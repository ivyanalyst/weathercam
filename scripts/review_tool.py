import argparse
import itertools
import json
import random
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from weathercam.data.veri import VeriReader, read_names
from weathercam.review.verdicts import BOX_ELEMENTS, REASONS, Record, append_record, judge, latest, read_log

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Review</title>
<style>
body{font-family:sans-serif;margin:0;display:flex;height:100vh}
#left{flex:1;overflow:auto;background:#222}
#right{width:340px;padding:12px;overflow:auto;border-left:1px solid #888}
button,select,input{font-size:15px;margin:3px 3px 3px 0;padding:5px 8px}
#msg{color:#06c;min-height:1.4em}
#err{color:#c00;min-height:1.4em}
canvas{display:block;cursor:crosshair}
</style></head><body>
<div id="left"><canvas id="cv"></canvas></div>
<div id="right">
<div id="prog"></div>
<div>Zoom <select id="zoom"><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
<h3 id="prompt"></h3>
<div id="ctrl"></div>
<hr>
<div><button id="unsure">Unsure (u)</button></div>
<div><select id="reason"></select><button id="nd">Can't judge</button></div>
<div><input id="note" placeholder="note (optional)" style="width:95%"></div>
<div><button id="back">Back</button></div>
<div id="msg"></div><div id="err"></div>
</div>
<script>
const REASONS = __REASONS__;
const $ = id => document.getElementById(id);
let pos = 0, item = null, ei = 0, box = null, drag = null, total = 0;
const img = new Image();
REASONS.forEach(r => { const o = document.createElement('option'); o.value = r; o.textContent = r; $('reason').appendChild(o); });

function get(path) { return fetch(path).then(r => r.json()); }
function post(body) {
  return fetch('/api/verdict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}).then(r => r.json());
}
function scale() { return (860 / img.naturalWidth) * Number($('zoom').value); }
function asked() {
  const v = item.existing.vehicle_visible;
  return (item.source === 'veri' && v && v.verdict === 'wrong') ? item.elements.slice(0, 1) : item.elements;
}
function cur() { return asked()[ei]; }
function button(label, fn) { const b = document.createElement('button'); b.textContent = label; b.onclick = fn; return b; }

function sizeCanvas() {
  const cv = $('cv'), s = scale();
  cv.width = Math.round(img.naturalWidth * s);
  cv.height = Math.round(img.naturalHeight * s);
}
function rect(c, b, color, s) {
  c.lineWidth = 2; c.strokeStyle = color;
  c.strokeRect(b[0] * s, b[1] * s, (b[2] - b[0]) * s, (b[3] - b[1]) * s);
}
function redraw() {
  const cv = $('cv'), c = cv.getContext('2d'), s = scale();
  c.clearRect(0, 0, cv.width, cv.height);
  c.drawImage(img, 0, 0, cv.width, cv.height);
  const e = cur();
  if (e && e.kind === 'box' && e.model_box && !box) rect(c, e.model_box, '#2d8cff', s);
  if (box) rect(c, box, '#ff9500', s);
}
$('cv').onmousedown = ev => {
  const e = cur();
  if (!e || e.kind !== 'box') return;
  const s = scale();
  drag = [ev.offsetX / s, ev.offsetY / s];
};
$('cv').onmousemove = ev => {
  if (!drag) return;
  const s = scale(), x = ev.offsetX / s, y = ev.offsetY / s;
  box = [Math.min(drag[0], x), Math.min(drag[1], y), Math.max(drag[0], x), Math.max(drag[1], y)];
  redraw();
};
window.onmouseup = () => { drag = null; };
$('zoom').onchange = () => { sizeCanvas(); redraw(); };

function submitBox() {
  if (box) submit({action: 'answer', value: box.map(v => Math.round(v * 10) / 10)});
  else if (cur().model_box) submit({action: 'box_ok'});
}

function render() {
  const e = cur(), a = asked();
  $('prog').textContent = 'Item ' + (pos + 1) + ' of ' + total + ' (' + item.source + '), element ' + (ei + 1) + '/' + a.length;
  $('prompt').textContent = e.prompt;
  const ctrl = $('ctrl');
  ctrl.innerHTML = '';
  const ex = item.existing[e.name];
  if (ex) {
    const p = document.createElement('div');
    p.textContent = 'Already recorded: ' + ex.verdict + (ex.corrected !== null ? ' (' + JSON.stringify(ex.corrected) + ')' : '') + (ex.reason ? ' [' + ex.reason + ']' : '');
    ctrl.appendChild(p);
  }
  if (e.kind === 'choice') {
    if (e.options.length > 12) {
      const sel = document.createElement('select');
      e.options.forEach(o => { const op = document.createElement('option'); op.value = o.value; op.textContent = o.label; sel.appendChild(op); });
      ctrl.appendChild(sel);
      ctrl.appendChild(button('Submit', () => submit({action: 'answer', value: sel.value})));
    } else {
      e.options.forEach((o, i) => {
        ctrl.appendChild(button(o.label + (i < 10 ? ' (' + ((i + 1) % 10) + ')' : ''), () => submit({action: 'answer', value: o.value})));
      });
    }
    if (e.extra_nd) ctrl.appendChild(button(e.extra_nd.label, () => submit({action: 'nd', reason: e.extra_nd.reason})));
  } else if (e.kind === 'text') {
    const inp = document.createElement('input');
    inp.id = 'txt'; inp.value = e.model_text || ''; inp.style.width = '95%';
    inp.addEventListener('keydown', ev => { if (ev.key === 'Enter') submit({action: 'answer', value: inp.value}); });
    ctrl.appendChild(inp);
    ctrl.appendChild(button('Submit text (Enter)', () => submit({action: 'answer', value: inp.value})));
  } else {
    if (e.model_box) ctrl.appendChild(button('Box is right (Enter)', () => submit({action: 'box_ok'})));
    ctrl.appendChild(button('Submit drawn box (Enter)', () => { if (box) submitBox(); }));
    const hint = document.createElement('div');
    hint.textContent = 'Drag on the image to draw a box.';
    ctrl.appendChild(hint);
  }
}

async function submit(body) {
  const e = cur();
  body.pos = pos; body.element = e.name; body.note = $('note').value;
  const r = await post(body);
  if (r.error) { $('err').textContent = r.error; return; }
  $('err').textContent = '';
  $('msg').textContent = 'recorded: ' + r.verdict + ' (' + e.name + ')';
  $('note').value = '';
  item.existing[e.name] = {verdict: r.verdict, corrected: r.corrected, reason: r.reason};
  box = null;
  advance();
}
function advance() {
  const a = asked();
  if (ei + 1 < a.length) { ei++; render(); redraw(); return; }
  pos++;
  if (pos >= total) { finish(); return; }
  loadItem('first');
}
function finish() {
  $('prompt').textContent = 'All items of this session are recorded.';
  $('ctrl').innerHTML = '';
  $('prog').textContent = '';
}
async function loadItem(where) {
  item = await get('/api/item?pos=' + pos);
  if (item.error) { $('err').textContent = item.error; return; }
  total = item.total;
  const a = asked();
  if (where === 'last') ei = a.length - 1;
  else { const k = a.findIndex(e => !item.existing[e.name]); ei = k < 0 ? 0 : k; }
  box = null;
  img.onload = () => { sizeCanvas(); redraw(); };
  img.src = '/img?pos=' + pos + '&t=' + Date.now();
  render();
}
$('unsure').onclick = () => { if (item) submit({action: 'unsure'}); };
$('nd').onclick = () => { if (item) submit({action: 'nd', reason: $('reason').value}); };
$('back').onclick = async () => {
  if (!item) return;
  if (ei > 0) { ei--; box = null; render(); redraw(); }
  else if (pos > 0) { pos--; await loadItem('last'); }
};
document.addEventListener('keydown', ev => {
  const t = ev.target.tagName;
  if (t === 'INPUT' || t === 'SELECT' || t === 'TEXTAREA' || !item) return;
  const e = cur();
  if (!e) return;
  if (ev.key === 'u') submit({action: 'unsure'});
  else if (e.kind === 'choice' && e.options.length <= 12 && /^[0-9]$/.test(ev.key)) {
    const i = (Number(ev.key) + 9) % 10;
    if (e.options[i]) submit({action: 'answer', value: e.options[i].value});
  } else if (e.kind === 'box' && ev.key === 'Enter') submitBox();
});
async function start() {
  const s = await get('/api/state');
  total = s.total;
  if (s.first_incomplete === null) { finish(); return; }
  pos = s.first_incomplete;
  loadItem('first');
}
start();
</script></body></html>
"""
PAGE = PAGE.replace("__REASONS__", json.dumps(list(REASONS)))

PROMPTS = {
    "colour": "What colour is the vehicle?",
    "type": "What type of vehicle is it?",
    "brand": "Which make is the vehicle?",
    "view": "From which side is the vehicle seen?",
}


class App:
    def __init__(self, queue, log_path, source, seed, limit):
        self.log_path = Path(log_path)
        self.lock = threading.Lock()
        items = [i for i in queue["items"] if source in ("both", i["source"])]
        parts = {}
        for name in ("veri", "ccpd"):
            part = sorted((i for i in items if i["source"] == name), key=lambda i: i["image"])
            random.Random(seed + (name == "ccpd")).shuffle(part)
            parts[name] = part
        mixed = [x for pair in itertools.zip_longest(parts["veri"], parts["ccpd"]) for x in pair if x is not None]
        self.order = mixed[:limit] if limit > 0 else mixed
        paths = queue["paths"]
        self.ccpd_root = Path(paths["ccpd_root"]).resolve()
        self.veri = None
        self.options = {}
        if parts["veri"]:
            self.veri = VeriReader(paths["veri_zip"])
            colors, types = read_names(paths["veri_zip"])
            makes = [str(m) for m in np.load(queue["provenance"]["head_brand_b16"]["path"])["makes"]]
            self.options = {
                "colour": list(colors.values()),
                "type": list(types.values()),
                "view": ["front", "rear", "side"],
                "brand": [m for m in makes if m != "unknown"],
            }

    def close(self):
        if self.veri is not None:
            self.veri.close()

    def lookup(self):
        return latest(read_log(self.log_path))

    def asked(self, item, lat):
        names = list(item["elements"])
        if item["source"] == "veri":
            r = lat.get(("veri", item["image"], "vehicle_visible"))
            if r is not None and r.verdict == "wrong":
                return names[:1]
        return names

    def complete(self, item, lat):
        return all((item["source"], item["image"], n) in lat for n in self.asked(item, lat))

    def state(self):
        lat = self.lookup()
        done = [self.complete(i, lat) for i in self.order]
        first = next((p for p, d in enumerate(done) if not d), None)
        return {"total": len(self.order), "complete": sum(done), "first_incomplete": first}

    def check_pos(self, value):
        pos = int(value)
        if not 0 <= pos < len(self.order):
            raise ValueError(f"no item at position {pos}")
        return pos

    def position(self, query):
        return self.check_pos(query["pos"][0])

    def spec(self, item, name):
        model = item["elements"][name]["model_value"]
        if name == "vehicle_visible":
            return {"name": name, "kind": "choice", "prompt": "Is a vehicle visible in the image?",
                    "options": [{"value": "vehicle", "label": "Vehicle visible"},
                                {"value": "no_vehicle", "label": "No vehicle visible"}]}
        if name in self.options:
            out = {"name": name, "kind": "choice", "prompt": PROMPTS[name],
                   "options": [{"value": o, "label": o} for o in self.options[name]]}
            if name == "brand":
                out["extra_nd"] = {"label": "Brand not in this list", "reason": "brand_not_in_list"}
            return out
        if name == "plate_text":
            if model is None:
                prompt = "Type the plate text you can read."
            else:
                prompt = "Is the reading right? Edit it to the true text."
            return {"name": name, "kind": "text", "model_text": model, "prompt": prompt}
        if name == "plate_box":
            prompt = "Drag a box around the number plate, or mark it not visible."
        elif model is None:
            prompt = "No box was found. Drag a box around the plate."
        else:
            prompt = "Is this box on the plate and does it cover all of it?"
        return {"name": name, "kind": "box", "model_box": model, "prompt": prompt}

    def item_json(self, pos):
        item = self.order[pos]
        lat = self.lookup()
        existing = {}
        for name in item["elements"]:
            r = lat.get((item["source"], item["image"], name))
            if r is not None:
                existing[name] = {"verdict": r.verdict, "corrected": r.corrected, "reason": r.reason}
        return {"pos": pos, "total": len(self.order), "source": item["source"],
                "elements": [self.spec(item, n) for n in item["elements"]], "existing": existing}

    def image(self, pos):
        item = self.order[pos]
        if item["source"] == "veri":
            return self.veri.read_image("test", item["image"])
        path = (self.ccpd_root / item["image"]).resolve()
        if self.ccpd_root not in path.parents:
            raise ValueError("image path outside the CCPD folder")
        return path.read_bytes()

    def make_record(self, item, name, body):
        model = item["elements"][name]["model_value"]
        note = body.get("note")
        if isinstance(note, str):
            note = note.strip() or None
        elif note is not None:
            raise ValueError("note must be a string")
        common = dict(source=item["source"], image=item["image"], split=item["split"], element=name,
                      model_value=model, timestamp=datetime.now().isoformat(timespec="seconds"), note=note)
        action = body.get("action")
        if action == "unsure":
            return Record(verdict="unsure", **common)
        if action == "nd":
            return Record(verdict="not_determinable", reason=body.get("reason"), **common)
        if action == "box_ok":
            if name not in BOX_ELEMENTS or model is None:
                raise ValueError("box_ok needs a box element with a model box")
            return Record(verdict="correct", **common)
        if action == "answer":
            value = body.get("value")
            if name in BOX_ELEMENTS:
                return Record(verdict="provided" if model is None else "wrong", corrected=value, **common)
            if isinstance(value, str):
                value = value.strip()
            if model is None:
                return Record(verdict="provided", corrected=value, **common)
            verdict, corrected = judge(name, model, value)
            return Record(verdict=verdict, corrected=corrected, **common)
        raise ValueError(f"unknown action: {action!r}")

    def record(self, body):
        pos = self.check_pos(body.get("pos"))
        item = self.order[pos]
        name = body.get("element")
        if name not in item["elements"]:
            raise ValueError(f"unknown element: {name!r}")
        rec = self.make_record(item, name, body)
        with self.lock:
            append_record(self.log_path, rec)
        return {"ok": True, "verdict": rec.verdict, "corrected": rec.corrected, "reason": rec.reason}


class Handler(BaseHTTPRequestHandler):
    app = None

    def log_message(self, *args):
        pass

    def reply(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def reply_json(self, data, code=200):
        self.reply(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/":
                self.reply(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self.reply_json(self.app.state())
            elif url.path == "/api/item":
                self.reply_json(self.app.item_json(self.app.position(query)))
            elif url.path == "/img":
                self.reply(200, self.app.image(self.app.position(query)), "image/jpeg")
            else:
                self.reply(404, b"not found", "text/plain")
        except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
            self.reply_json({"error": str(error)}, 400)

    def do_POST(self):
        if urlparse(self.path).path != "/api/verdict":
            self.reply(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100000:
                raise ValueError("bad request size")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
            self.reply_json(self.app.record(body))
        except (ValueError, KeyError, IndexError, TypeError) as error:
            self.reply_json({"error": str(error)}, 400)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", default="outputs/review/queue_v1.json")
    p.add_argument("--log", default="outputs/review/log_v1.jsonl")
    p.add_argument("--source", choices=("both", "veri", "ccpd"), default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="only the first N items of the order (0 = all)")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


def main():
    args = parse_args()
    if args.limit < 0:
        sys.exit("--limit must be >= 0")
    queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    app = App(queue, args.log, args.source, args.seed, args.limit)
    Handler.app = app
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    state = app.state()
    print("items", state["total"], "complete", state["complete"], "log", args.log)
    print("open http://127.0.0.1:%d/   (Ctrl+C to stop)" % args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
