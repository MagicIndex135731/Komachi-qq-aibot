"""Execute the patched upstream helpers with synthetic cookies and transports."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / 'infra/wsl/scripts/patch_snowluma_qzone.cjs'

# Relevant upstream helper shapes; no account state or third-party runtime copy.
SOURCE = '''
async function getCookies(bridge, domain) {
    const data = await RequestUtil.HttpsGetCookies("https://synthetic.invalid");
    if (!data["p_skey"] || data["p_skey"].length === 0) try {
        const pskey = (await getPSkey(bridge, [domain])).domainPskeyMap.get(domain);
        if (pskey) data["p_skey"] = pskey;
    } catch {
        return data;
    }
    return data;
}
async function getSKey(bridge) { return "unused"; }
async function requestMsgList(route, targetUin) {
    return RequestUtil.HttpGetText(route.url, "GET", "", route.headers);
}
async function getQzoneFeeds(cookieObject) {
    const url = "https://synthetic.invalid/feed";
    const text = await RequestUtil.HttpGetText(url, "GET", "", { Cookie: cookieToString(cookieObject) });
    const data = parseQzoneCallback(text);
    return data;
}
'''


def run_node(script, tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node runtime required for executable bridge compatibility checks')
    value = subprocess.run([node, '-e', script, str(PATCH)], input=SOURCE,
        capture_output=True, text=True, encoding='utf8', cwd=tmp_path, timeout=10)
    assert value.returncode == 0, value.stderr
    return json.loads(value.stdout)


def test_cookie_identity_fallback_scope_and_read_limits(tmp_path):
    result = run_node('''
const {patchSource}=require(process.argv[1]);const fs=require('node:fs');
const source=fs.readFileSync(0,'utf8');let jar={},native={},failure=false;const reads=[];
const transport={HttpsGetCookies:async()=>({...jar}),HttpGetText:async(...args)=>{reads.push(args);return '{}'}};
const pskey=async()=>{if(failure)throw Error('synthetic');return {domainPskeyMap:new Map(Object.entries(native))}};
const api=new Function('RequestUtil','getPSkey','cookieToString','parseQzoneCallback',patchSource(source)+';return {getCookies,requestMsgList,getQzoneFeeds};')(transport,pskey,()=>'',JSON.parse);
(async()=>{
 const bridge={identity:{uin:'900000099'}};const results={};
 jar={p_skey:'synthetic'};results.partial=await api.getCookies(bridge,'qzone.qq.com');
 jar={};native={'qzone.qq.com':'native-synthetic'};results.native=await api.getCookies(bridge,'qzone.qq.com');
 jar={skey:'synthetic',uin:'o000900000099',p_uin:'o900000099'};failure=true;
 results.catch=await api.getCookies(bridge,'qzone.qq.com');
 jar={p_skey:'synthetic',p_uin:'o900000101'};
 try{await api.getCookies(bridge,'qzone.qq.com')}catch(e){results.foreign=e.message}
 jar={};native={};failure=false;
 try{await api.getCookies(bridge,'qzone.qq.com')}catch(e){results.noauth=e.message}
 results.other=await api.getCookies(bridge,'qun.qq.com');
 await api.requestMsgList({url:'https://synthetic.invalid/list',headers:{}},'900000101');
 await api.getQzoneFeeds({p_skey:'synthetic'});
 results.limits=reads.map(args=>args[4]);results.idempotent=patchSource(patchSource(source))===patchSource(source);
 console.log(JSON.stringify(results));
})().catch(e=>{console.error(e);process.exitCode=1});
''', tmp_path)
    assert result['partial']['uin'] == result['native']['p_uin'] == 'o900000099'
    assert result['catch']['uin'] == 'o000900000099'
    assert result['foreign'] == 'Qzone cookie account mismatch'
    assert result['noauth'] == 'Qzone authentication cookie unavailable'
    assert result['other'] == {}
    assert result['limits'] == [{'timeoutMs': 5000, 'maxResponseBytes': 2097152}] * 2
    assert result['idempotent']


def test_unsupported_upstream_source_is_rejected_without_modification(tmp_path):
    result = run_node('''
const {patchSource}=require(process.argv[1]);let rejected=false;
try{patchSource('async function getCookies(){}')}catch(e){rejected=true}
console.log(JSON.stringify({rejected}));
''', tmp_path)
    assert result['rejected']
