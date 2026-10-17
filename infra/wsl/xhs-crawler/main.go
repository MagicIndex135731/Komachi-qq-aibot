// Container-only adapter over xpzouying/xiaohongshu-mcp (Apache-2.0).
// No listening socket and no publishing, comments, likes, or account mutations.
package main

import (
    "bytes"
    "context"
    "encoding/base64"
    "encoding/json"
    "io"
    "os"
    "path/filepath"
    "regexp"
    "strings"
    "sync"
    "syscall"
    "time"

    "github.com/go-rod/rod"
    "github.com/go-rod/rod/lib/launcher"
    "github.com/go-rod/rod/lib/proto"
    "github.com/sirupsen/logrus"
    "github.com/xpzouying/xiaohongshu-mcp/xiaohongshu"
)

type Request struct {
    Action string `json:"action"`
    Query string `json:"query"`
    ID string `json:"id"`
    Token string `json:"token"`
    Limit int `json:"limit"`
    Sort string `json:"sort"`
    MediaType string `json:"media_type"`
}

func emit(value interface{}) {
    body,err:=json.Marshal(value)
    if err!=nil || len(body)>512*1024 {body=[]byte(`{"ok":false,"error":"response_size"}`)}
    _,_=os.Stdout.Write(append(body,'\n'))
}
func failure(code string) { emit(map[string]interface{}{"ok":false,"error":code}) }

func classify(page *rod.Page) string {
    value, err := page.Timeout(3*time.Second).Eval(`() => ({url:location.href,text:(document.body?.innerText||'').slice(0,6000),login:!!document.querySelector('.login-container'),captcha:!!document.querySelector('[class*="captcha"], [id*="captcha"]')})`)
    if err != nil { return "observation_failed" }
    raw := value.Value.JSON("", "")
    if strings.Contains(raw,"captcha\":true") || strings.Contains(raw,"安全验证") || strings.Contains(raw,"访问验证") { return "challenge_required" }
    if strings.Contains(raw,"访问频次") || strings.Contains(raw,"操作频繁") { return "rate_limited" }
    if strings.Contains(raw,"私密笔记") || strings.Contains(raw,"仅作者可见") || strings.Contains(raw,"仅自己可见") || strings.Contains(raw,"因用户设置") { return "permission_denied" }
    if strings.Contains(raw,"页面不见了") || strings.Contains(raw,"笔记不存在") || strings.Contains(raw,"已被删除") || strings.Contains(raw,"暂时无法浏览") { return "note_unavailable" }
    return ""
}

func run() {
    logrus.SetOutput(io.Discard)
    allowedEnv:=map[string]string{}
    for _,key:=range []string{"PATH","LANG","TZ"} {if value:=os.Getenv(key);value!="" {allowedEnv[key]=value}}
    os.Clearenv()
    for key,value:=range allowedEnv {_=os.Setenv(key,value)}
    _=os.Setenv("HOME","/nonexistent")
    var req Request
    input,readErr:=io.ReadAll(io.LimitReader(os.Stdin,8193))
    if readErr!=nil || len(input)>8192 {failure("invalid_request");return}
    dec := json.NewDecoder(bytes.NewReader(input)); dec.DisallowUnknownFields()
    if err:=dec.Decode(&req); err!=nil { failure("invalid_request");return }
    var extra interface{}
    if dec.Decode(&extra)!=io.EOF {failure("invalid_request");return}
    if req.Action!="login" && req.Action!="status" && req.Action!="search" && req.Action!="inspect" {failure("invalid_action");return}
    if req.Action=="search" && (len([]rune(req.Query))<1 || len([]rune(req.Query))>120 || req.Limit<1 || req.Limit>12) {failure("invalid_request");return}
    if req.Action=="search" && ((req.Sort!="" && req.Sort!="latest" && req.Sort!="relevance") || (req.MediaType!="" && req.MediaType!="image" && req.MediaType!="video")) {failure("invalid_request");return}
    if req.Action=="inspect" && (!regexp.MustCompile(`^[a-f0-9]{24}$`).MatchString(req.ID) || len(req.Token)>1024 || !regexp.MustCompile(`^[A-Za-z0-9_+/=-]*$`).MatchString(req.Token)) {failure("invalid_target");return}
    duration:=80*time.Second
    if req.Action=="login" {duration=5*time.Minute}
    ctx,cancel:=context.WithTimeout(context.Background(),duration);defer cancel()
    dir:="/workspace/data/xhs-crawler"
    if err:=os.MkdirAll(dir,0700);err!=nil {failure("session_storage_failed");return}
    _=os.Chmod(dir,0700)
    lock,err:=os.OpenFile(filepath.Join(dir,"worker.lock"),os.O_CREATE|os.O_RDWR,0600)
    if err!=nil {failure("session_storage_failed");return};defer lock.Close()
    if syscall.Flock(int(lock.Fd()),syscall.LOCK_EX|syscall.LOCK_NB)!=nil {failure("crawler_busy");return}
    defer syscall.Flock(int(lock.Fd()),syscall.LOCK_UN)
    // A per-process temporary browser profile; durable cookies remain 0600 in
    // the named data volume and never cross stdout or the model tool channel.
    profile,err:=os.MkdirTemp("","xiaomachi-xhs-");if err!=nil {failure("browser_unavailable");return};defer os.RemoveAll(profile)
    l:=launcher.New().Bin("/opt/xhs/chrome/chrome").Headless(true).NoSandbox(true).UserDataDir(profile).Set("lang","zh-CN").Set("disable-dev-shm-usage").Set("disable-gpu")
    if req.Action!="login" {l.Set("blink-settings","imagesEnabled=false")}
    control,err:=l.Launch();if err!=nil {failure("browser_unavailable");return}
    defer l.Cleanup()
    b:=rod.New().ControlURL(control).Context(ctx)
    if err=b.Connect();err!=nil {failure("browser_unavailable");return};defer b.Close()
    session:=filepath.Join(dir,"session.json")
    if raw,err:=os.ReadFile(session);err==nil {
        var stored []*proto.NetworkCookieParam
        if json.Unmarshal(raw,&stored)!=nil {failure("session_invalid");return}
        if err=b.SetCookies(stored);err!=nil {failure("session_invalid");return}
    } else if req.Action!="login" {
        if os.IsNotExist(err) {failure("login_required")} else {failure("session_invalid")};return
    }
    page,err:=b.Page(proto.TargetCreateTarget{});if err!=nil {failure("browser_unavailable");return}
    defer page.Close()
    var mu sync.Mutex
    requests,apiRequests,received,networkFailure:=0,0,float64(0),""
    stopNetwork:=func(code string){if networkFailure=="" {networkFailure=code;cancel()}}
    requestLimit:=600;if req.Action=="login" {requestLimit=1200}
    waitEvents:=page.EachEvent(func(e *proto.NetworkRequestWillBeSent){
        if !strings.HasPrefix(e.Request.URL,"https://") && !strings.HasPrefix(e.Request.URL,"http://") {return}
        mu.Lock();defer mu.Unlock();requests++
        if requests>requestLimit {stopNetwork("request_budget")}
        if (e.Type=="XHR" || e.Type=="Fetch") && strings.Contains(e.Request.URL,".xiaohongshu.com/api/") {
            apiRequests++;if apiRequests>180 {stopNetwork("api_request_budget")}
        }
    },func(e *proto.NetworkDataReceived){
        mu.Lock();defer mu.Unlock();received+=float64(e.DataLength)
        if received>32*1024*1024 {stopNetwork("response_size")}
    },func(e *proto.NetworkResponseReceived){
        // Stop on the main site's document refusal; a third-party image or
        // analytics failure is not evidence of an account access failure.
        mainDocument:=e.Type==proto.NetworkResourceTypeDocument && strings.HasPrefix(e.Response.URL,"https://www.xiaohongshu.com/")
        siteAPI:=(e.Type=="XHR" || e.Type=="Fetch") && strings.Contains(e.Response.URL,".xiaohongshu.com/api/")
        if mainDocument || siteAPI {
            mu.Lock();defer mu.Unlock()
            if e.Response.Status==429 {stopNetwork("rate_limited")}
            if e.Response.Status==401 || e.Response.Status==403 {stopNetwork("permission_denied")}
        }
    })
    go waitEvents()
    pageFailure:=func()string{
        mu.Lock();code:=networkFailure;mu.Unlock()
        if code!="" {return code};return classify(page)
    }
    fail:=func(code string){
        mu.Lock();counts:=map[string]interface{}{"http_requests":requests,"api_requests":apiRequests,"received_bytes":received};mu.Unlock()
        emit(map[string]interface{}{"ok":false,"error":code,"metrics":counts})
    }
    defer func(){if recover()!=nil {code:=pageFailure();if code=="" {code="acquisition_failed"};fail(code)}}()
    login:=xiaohongshu.NewLogin(page)
    if req.Action=="login" {
        qrCtx,qrCancel:=context.WithTimeout(ctx,40*time.Second)
        img,logged,err:=login.FetchQrcodeImage(qrCtx);qrCancel()
        if err!=nil {failure("login_unavailable");return}
        if !logged {
            if code:=pageFailure();code!="" {failure(code);return}
            comma:=strings.Index(img,",")
            if !strings.HasPrefix(img,"data:image/") || comma<0 {failure("login_qr_invalid");return}
            qr,err:=base64.StdEncoding.DecodeString(img[comma+1:]);if err!=nil || len(qr)>1024*1024 {failure("login_qr_invalid");return}
            if err=os.WriteFile(filepath.Join(dir,"login-qr.png"),qr,0600);err!=nil {failure("session_storage_failed");return}
            emit(map[string]interface{}{"ok":true,"status":"waiting_for_user_scan","expires_seconds":240})
            waitCtx,stop:=context.WithTimeout(ctx,240*time.Second);logged=login.WaitForLogin(waitCtx);stop()
        }
        if !logged {failure("login_timeout");return}
        values,err:=b.GetCookies();if err!=nil {failure("session_storage_failed");return}
        stored:=make([]*proto.NetworkCookieParam,0,len(values))
        for _,c:=range values {stored=append(stored,&proto.NetworkCookieParam{Name:c.Name,Value:c.Value,Domain:c.Domain,Path:c.Path,Secure:c.Secure,HTTPOnly:c.HTTPOnly,Expires:c.Expires,SameSite:c.SameSite})}
        raw,_:=json.Marshal(stored)
        if err=os.WriteFile(session+".next",raw,0600);err!=nil {failure("session_storage_failed");return}
        if err=os.Rename(session+".next",session);err!=nil {failure("session_storage_failed");return}
        emit(map[string]interface{}{"ok":true,"authenticated":true});return
    }
    if req.Action=="status" {
        logged,err:=login.CheckLoginStatus(ctx)
        if code:=pageFailure();code!="" {fail(code);return}
        if err!=nil || !logged {failure("login_required");return}
        emit(map[string]interface{}{"ok":true,"authenticated":true});return
    }
    loggedPage:=func()bool{exists,_,err:=page.Timeout(3*time.Second).Has(`.main-container .user .link-wrapper .channel`);return err==nil && exists}
    if req.Action=="search" {
        sort:="综合";if req.Sort=="latest" {sort="最新"} else if req.Sort!="" && req.Sort!="relevance" {failure("invalid_request");return}
        kind:="不限";if req.MediaType=="image" {kind="图文"} else if req.MediaType=="video" {kind="视频"} else if req.MediaType!="" {failure("invalid_request");return}
        rows,err:=xiaohongshu.NewSearchAction(page).Search(ctx,req.Query,xiaohongshu.FilterOption{SortBy:sort,NoteType:kind})
        if code:=pageFailure();code!="" {fail(code);return}
        if !loggedPage() {failure("login_required");return}
        if err!=nil {failure("search_failed");return}
        if len(rows)>req.Limit {rows=rows[:req.Limit]}
        emit(map[string]interface{}{"ok":true,"data":rows,"authenticated":true,"coverage":"visible_candidates_only"});return
    }
    detail,err:=xiaohongshu.NewFeedDetailAction(page).GetFeedDetail(ctx,req.ID,req.Token,false,xiaohongshu.DefaultCommentLoadConfig())
    if code:=pageFailure();code!="" {fail(code);return}
    if !loggedPage() {failure("login_required");return}
    if err!=nil {failure("note_unavailable");return}
    if detail.Note.NoteID!=req.ID {failure("identity_mismatch");return}
    emit(map[string]interface{}{"ok":true,"data":detail.Note,"authenticated":true})
}

func main(){run()}
