// PRISM API client. The dashboard is served by the API itself, so requests are same-origin;
// opening index.html from disk falls back to the default local server.
export const API = location.protocol.startsWith("http") ? "" : "http://127.0.0.1:8000";

async function fail(res) {
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body && body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch (_) { /* not JSON */ }
  return new Error(detail);
}

function bust(path) {
  return `${API}${path}${path.includes("?") ? "&" : "?"}_=${Date.now()}`;
}

export async function get(path) {
  const res = await fetch(bust(path));
  if (!res.ok) throw await fail(res);
  return res.json();
}

export async function getBinary(path) {
  const res = await fetch(bust(path));
  if (!res.ok) throw await fail(res);
  return res.arrayBuffer();
}

export async function exists(path) {
  try {
    const res = await fetch(bust(path), { method: "HEAD" });
    return res.ok;
  } catch (_) {
    return false;
  }
}

export async function post(path, body = {}) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await fail(res);
  return res.json();
}

export async function del(path) {
  const res = await fetch(`${API}${path}`, { method: "DELETE" });
  if (!res.ok) throw await fail(res);
  return res.json();
}

// multipart upload with progress (fetch cannot report upload progress)
export function upload(path, form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}${path}`);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText || "{}"); } catch (_) { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.detail || `${xhr.status} ${xhr.statusText}`));
    };
    xhr.onerror = () => reject(new Error("Network error while uploading"));
    xhr.send(form);
  });
}

export function url(path) {
  return `${API}/${path.replace(/^\//, "")}`;
}
