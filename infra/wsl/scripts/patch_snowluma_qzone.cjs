"use strict";
// Compatibility for the pinned SnowLuma runtime. No credentials are retained.
const fs = require("node:fs");
const path = require("node:path");
const marker = "// xiaomachi-qzone-cookie-v1";
const complete = `${marker}
function completeQzoneCookies(data, account, domain) {
    if (domain !== "qzone.qq.com") return data;
    const id = String(account);
    if (!/^[1-9][0-9]{0,19}$/.test(id)) throw new Error("Qzone cookie account invalid");
    if (![data.p_skey, data.skey].some(v => typeof v === "string" && v.length > 0))
        throw new Error("Qzone authentication cookie unavailable");
    for (const name of ["uin", "p_uin"]) {
        if (data[name]) {
            if (typeof data[name] !== "string" || !/^o[0-9]{1,20}$/.test(data[name])
                    || BigInt(data[name].slice(1)) !== BigInt(id))
                throw new Error("Qzone cookie account mismatch");
        } else data[name] = "o" + id;
    }
    return data;
}
`;

function replaceOnce(source, before, after) {
    if (source.split(before).length !== 2) throw new Error("Unsupported SnowLuma Qzone source");
    return source.replace(before, after);
}

function patchSource(source) {
    if (source.includes(marker)) return source;
    const start = source.indexOf("async function getCookies(bridge, domain) {");
    const end = source.indexOf("async function getSKey(bridge)", start);
    if (start < 0 || end < start) throw new Error("Unsupported SnowLuma cookie helper");
    let cookie = source.slice(start, end);
    if (cookie.split("return data;").length !== 3) throw new Error("Unsupported SnowLuma cookie returns");
    cookie = cookie.replaceAll("return data;", "return completeQzoneCookies(data, bridge.identity.uin, domain);");
    source = source.slice(0, start) + complete + cookie + source.slice(end);
    source = replaceOnce(source,
        'RequestUtil.HttpGetText(route.url, "GET", "", route.headers)',
        'RequestUtil.HttpGetText(route.url, "GET", "", route.headers, { timeoutMs: 5000, maxResponseBytes: 2097152 })');
    const feedStart = source.indexOf("async function getQzoneFeeds(");
    const feedEnd = source.indexOf("const data = parseQzoneCallback(text);", feedStart);
    if (feedStart < 0 || feedEnd < feedStart) throw new Error("Unsupported SnowLuma feed helper");
    const feed = replaceOnce(source.slice(feedStart, feedEnd),
        'RequestUtil.HttpGetText(url, "GET", "", { Cookie: cookieToString(cookieObject) })',
        'RequestUtil.HttpGetText(url, "GET", "", { Cookie: cookieToString(cookieObject) }, { timeoutMs: 5000, maxResponseBytes: 2097152 })');
    return source.slice(0, feedStart) + feed + source.slice(feedEnd);
}

if (require.main === module) {
    const file = path.join(process.argv[2] || "/app/runtime", "index.mjs");
    const original = fs.readFileSync(file, "utf8");
    const patched = patchSource(original); // Validate every anchor before writing.
    if (patched !== original) fs.writeFileSync(file, patched);
    process.stdout.write("snowluma_qzone_compat_ready\n");
}
module.exports = { patchSource };
