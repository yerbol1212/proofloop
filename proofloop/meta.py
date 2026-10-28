"""Small Meta Responses API adapter; secrets and owned media stay local.

Wire contracts checked against Meta's official docs on 2026-10-26:
https://dev.meta.ai/docs/structured-output
https://dev.meta.ai/docs/sam/segmenting
https://dev.meta.ai/docs/sam/reading-segmentation

No offline/model-shaped result is substituted for a failed live request.
The USD ledger is an estimate, not a provider-enforced billing limit.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

BASE_URL = "https://api.meta.ai/v1"
SPARK_MODEL = "muse-spark-1.3"
SAM_MODEL = "sam-3.1"
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def _reported_usage(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _reported_tokens(value: Any) -> int:
    return value if type(value) is int and value >= 0 else 0


class MetaAPIError(RuntimeError):
    """Safe, deliberately terse error which never embeds provider message bodies."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None,
                 response: dict[str, Any] | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.response = response  # Local diagnostics only; omitted from error text.


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the bearer token to a redirected host.
        raise MetaAPIError("Meta API redirected unexpectedly; request stopped", status=code)


def load_env_file(path: str | Path = ".env.local") -> None:
    """Read a deliberately small KEY=value format without executing shell code.

    Existing process variables win. Only META_API_KEY is imported; quote removal
    is literal. This function never prints file content or the variable value.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip() == "META_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault("META_API_KEY", value)


def _image_data(path: str | Path) -> tuple[str, dict[str, Any]]:
    image_path = Path(path)
    data = image_path.read_bytes()
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image must contain 1 byte to 5 MiB")
    # Check file signatures; an arbitrary extension is not sufficient.
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError("Use a PNG, JPEG, or WebP image with a valid file signature")
    return (
        f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}",
        {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "mime": mime},
    )


def output_text(response: dict[str, Any]) -> str:
    """Extract Responses text lanes without treating reasoning as an answer."""
    pieces = []
    output = response.get("output", [])
    if not isinstance(output, list):
        raise MetaAPIError("Malformed Meta output collection", code="invalid_output")
    for item in output:
        if not isinstance(item, dict):
            raise MetaAPIError("Malformed Meta output item", code="invalid_output")
        if item.get("type") != "message":
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            raise MetaAPIError("Malformed Meta message content", code="invalid_output")
        for part in content:
            if not isinstance(part, dict):
                raise MetaAPIError("Malformed Meta message part", code="invalid_output")
            if part.get("type") == "refusal":
                raise MetaAPIError("Meta model declined this request", code="model_refusal")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                pieces.append(part["text"])
    # Inspect refusal parts before accepting any convenience output_text field.
    return response["output_text"] if isinstance(response.get("output_text"), str) else "".join(pieces)


def _sam_decoder() -> tuple[Any, Any, Any]:
    try:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster, InvalidSegmentationMaskError
    except (ImportError, TypeError, SyntaxError):
        raise MetaAPIError(
            "Official SAM mask parser unavailable. Use Python >=3.10 and install meta-sam-parser==0.1.0",
            code="sam_parser_unavailable",
        ) from None
    return SegmentationMask, decode_mask_to_raster, InvalidSegmentationMaskError


def decode_sam_masks(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decode complete masks using Meta's pinned official parser, never a guessed codec.

    The binary raster is stored as base64 of row-major 0/1 bytes, not an image
    file. Geometry-only callers can explicitly skip this optional dependency.
    Project-defined raster limits run before the official unbounded decoder.
    """
    Mask, decode, InvalidMask = _sam_decoder()
    result = []
    total_area = 0
    for record in objects:
        mask = record["mask"]
        area = mask["height"] * mask["width"]
        total_area += area
        if area > 16_000_000 or total_area > 32_000_000:
            raise MetaAPIError("SAM raster exceeds local decode limit", code="sam_raster_limit")
        try:
            raster = decode(Mask(encoding=mask["encoding"], payload=mask["encoded_payload"],
                                 width=mask["width"], height=mask["height"]))
        except InvalidMask:
            raise MetaAPIError("Official SAM parser rejected the encoded mask", code="invalid_sam_mask") from None
        if len(raster) != area or any(pixel not in {0, 1} for pixel in raster):
            raise MetaAPIError("Official SAM decoder returned an invalid raster", code="invalid_sam_raster")
        decoded = dict(mask, decoded=True, raster_format="base64_row_major_uint8_0_or_1",
                       raster_base64=base64.b64encode(raster).decode("ascii"),
                       foreground_pixels=sum(raster), raster_sha256=hashlib.sha256(raster).hexdigest())
        result.append(dict(record, mask=decoded))
    return result


def validate_json(value: Any, schema: dict[str, Any], location: str = "$") -> None:
    """Validate the small declarative schema subset used by this prototype.

    Refuse unsupported keywords instead of claiming general JSON Schema support.
    Validation is independent of the model's strict-output request.
    """
    allowed = {"type", "properties", "required", "additionalProperties", "items", "enum",
               "description", "minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength"}
    if set(schema) - allowed:
        raise ValueError(f"Unsupported schema keyword at {location}")
    kind = schema.get("type")
    match = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if kind not in match or not match[kind]:
        raise ValueError(f"Schema type mismatch at {location}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"Schema enum mismatch at {location}")
    if kind == "object":
        props = schema.get("properties", {})
        if not set(schema.get("required", [])).issubset(value):
            raise ValueError(f"Missing required field at {location}")
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise ValueError(f"Unexpected field at {location}")
        for key, item in value.items():
            if key in props:
                validate_json(item, props[key], f"{location}.{key}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            raise ValueError(f"Array size mismatch at {location}")
        for index, item in enumerate(value):
            validate_json(item, schema["items"], f"{location}[{index}]")
    elif kind in {"integer", "number"}:
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            raise ValueError(f"Number outside bounds at {location}")
    elif kind == "string":
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", len(value)):
            raise ValueError(f"String size mismatch at {location}")


_FRAME = re.compile(r"<(\d+)f>")
_RECORD = re.compile(
    r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)"
    r"(?:;c=([0-9]+(?:\.[0-9]+)?))?\|>"
    r"<\|mask;x=0;y=0(?:;c=([0-9]+(?:\.[0-9]+)?))?;data=(\d+),(\d+),([~!])([^\r\n]*?)\|>"
)


def parse_sam_boxes(text: str) -> list[dict[str, Any]]:
    """Parse documented SAM record geometry and preserve encoded masks verbatim.

    Empty output means zero detections, never proof of physical absence. Boxes
    use inclusive right/bottom pixel coordinates. Masks remain encoded and must
    be decoded with Meta's parser before any mask-based measurements are made.
    Sparse/nonascending source frames and object IDs are preserved.
    """
    if not text.strip():
        return []
    markers = list(_FRAME.finditer(text))
    if not markers or text[:markers[0].start()].strip():
        raise ValueError("Invalid SAM frame marker")
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index, marker in enumerate(markers):
        frame = int(marker.group(1))
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        block = text[marker.end():end].strip()
        if not block:
            raise ValueError("SAM frame marker has no object record")
        position = 0
        while position < len(block):
            match = _RECORD.match(block, position)
            if not match:
                raise ValueError("Invalid SAM box/mask record")
            oid, x1, y1, x2, y2, width, height = map(int, match.groups()[:7])
            box_conf, mask_conf, mask_h, mask_w, encoding, payload = match.groups()[7:]
            if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
                raise ValueError("SAM box outside source frame")
            if not payload or int(mask_h) < 1 or int(mask_w) < 1:
                raise ValueError("Empty SAM mask")
            if box_conf != mask_conf:
                raise ValueError("SAM box/mask confidence mismatch")
            if box_conf is not None and not 0 <= float(box_conf) <= 1:
                raise ValueError("SAM confidence outside [0,1]")
            if (frame, oid) in seen:
                raise ValueError("Duplicate SAM frame/object ID")
            seen.add((frame, oid))
            records.append({
                "frame": frame, "object_id": oid,
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": width, "height": height},
                "confidence": float(box_conf) if box_conf is not None else None,
                "mask": {"height": int(mask_h), "width": int(mask_w),
                         "encoding": "one_bit" if encoding == "!" else "lossless",
                         # Meta's public decoder expects the marker as the
                         # first payload character as well as the encoding ID.
                         "encoded_payload": encoding + payload, "decoded": False},
            })
            position = match.end()
            if position < len(block):
                if block[position] != ",":
                    raise ValueError("Invalid SAM record separator")
                position += 1
                if position == len(block):
                    raise ValueError("Trailing SAM record separator")
    return records


class MetaClient:
    """Server-side-only client with bounded attempts and redacted error handling."""

    def __init__(self, api_key: str, *, timeout: float = 90, max_requests: int = 8,
                 retries: int = 0, opener: Callable[..., Any] | None = None,
                 sleeper: Callable[[float], None] = time.sleep):
        if not api_key or any(c.isspace() for c in api_key):
            raise ValueError("META_API_KEY must be present and contain no whitespace")
        if max_requests < 1 or not 0 <= retries <= 2 or not 0 < timeout <= 180:
            raise ValueError("Invalid request/timeout/retry limit")
        self._api_key = api_key
        self.timeout = timeout
        self.max_requests = max_requests
        self.retries = retries
        self._opener = opener or build_opener(_NoRedirect()).open
        self._sleep = sleeper
        self.ledger: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls, **kwargs: Any) -> "MetaClient":
        key = os.environ.get("META_API_KEY", "")
        if not key:
            raise MetaAPIError("META_API_KEY is missing; set it in the environment or ignored .env.local")
        return cls(key, **kwargs)

    def _request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint not in {"models", "responses"}:
            raise ValueError("Unsupported Meta endpoint")
        encoded = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        request = Request(f"{BASE_URL}/{endpoint}", data=encoded, method=method, headers={
            "Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
            "Accept": "application/json", "User-Agent": "ProofLoop/0.1",
        })
        for attempt in range(self.retries + 1):
            if len(self.ledger) >= self.max_requests:
                raise MetaAPIError("Local request-attempt budget reached", code="local_request_limit")
            started = time.monotonic()
            entry: dict[str, Any] = {"endpoint": endpoint, "model": (payload or {}).get("model"),
                                     "attempt": attempt + 1, "status": "started"}
            self.ledger.append(entry)
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    # A bounded response prevents unbounded local allocations.
                    body = response.read(32 * 1024 * 1024 + 1)
                    if len(body) > 32 * 1024 * 1024:
                        raise MetaAPIError("Meta response exceeded the local size limit")
                    result = json.loads(body)
                if not isinstance(result, dict):
                    raise MetaAPIError("Meta response was not a JSON object")
                # An incomplete generation can contain billable reasoning
                # tokens. Retain its usage before rejecting it as evidence.
                entry["usage"] = _reported_usage(result.get("usage"))
                if result.get("error") or (endpoint == "responses" and result.get("status") != "completed"):
                    raise MetaAPIError("Meta response did not complete successfully", code="incomplete_response",
                                       response=result)
                entry["status"] = "ok"
                return result
            except HTTPError as error:
                code = None
                error_body = None
                try:
                    parsed_error = json.loads(error.read(64 * 1024))
                    if isinstance(parsed_error, dict):
                        error_body = parsed_error
                        if "usage" in error_body:
                            entry["usage"] = _reported_usage(error_body["usage"])
                    candidate = (error_body or {}).get("error", {}).get("code")
                    # Never include body/message text; it can echo request content.
                    if (isinstance(candidate, str) and re.fullmatch(r"[a-z_]{1,64}", candidate)
                            and self._api_key not in candidate):
                        code = candidate
                except (ValueError, AttributeError, TypeError):
                    pass
                entry.update(status="http_error", http_status=error.code, error_code=code)
                if error.code == 429 and attempt < self.retries:
                    self._sleep(min(2 ** attempt, 4))
                    continue
                raise MetaAPIError(f"Meta API HTTP {error.code}" + (f" ({code})" if code else ""),
                                   status=error.code, code=code, response=error_body) from None
            except (URLError, TimeoutError, OSError) as error:
                # Do not retry ambiguous connection failures: the server may bill
                # an inference that completed before the client lost connection.
                entry.update(status="network_error", error_type=type(error).__name__)
                raise MetaAPIError("Meta API connection failed; no automatic retry", code="network_error") from None
            except (json.JSONDecodeError, UnicodeDecodeError):
                entry.update(status="invalid_json")
                raise MetaAPIError("Meta API returned invalid JSON", code="invalid_json") from None
            except MetaAPIError:
                entry["status"] = "invalid_response"
                raise
            finally:
                entry["latency_seconds"] = round(time.monotonic() - started, 3)
        raise AssertionError("Request retry loop exhausted")

    def list_models(self) -> list[str]:
        result = self._request("GET", "models")
        models = result.get("data")
        if not isinstance(models, list):
            raise MetaAPIError("Meta models response omitted its data list")
        return [entry["id"] for entry in models if isinstance(entry, dict) and isinstance(entry.get("id"), str)]

    def spark_json(self, prompt: str, schema: dict[str, Any], *, name: str = "proofloop",
                   image_path: str | Path | None = None, image_paths: Sequence[str | Path] | None = None,
                   max_output_tokens: int = 1200) -> dict[str, Any]:
        if not 16 <= max_output_tokens <= 4096 or len(prompt.encode()) > 32 * 1024:
            raise ValueError("Spark prompt/token cap exceeded")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("Invalid JSON schema name")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        if image_path is not None and image_paths is not None:
            raise ValueError("Choose image_path or image_paths, not both")
        if image_paths is not None and (isinstance(image_paths, (str, bytes, Path))
                                       or not isinstance(image_paths, (list, tuple))
                                       or not 1 <= len(image_paths) <= 8):
            raise ValueError("Supply one to eight image paths")
        sources = []
        for path in ([image_path] if image_path is not None else image_paths or []):
            image_url, image_source = _image_data(path)
            sources.append(image_source)
            if sum(item["bytes"] for item in sources) > 20 * 1024 * 1024:
                raise ValueError("Spark images exceed the combined 20 MiB cap")
            content.append({"type": "input_image", "image_url": image_url})
        result = self._request("POST", "responses", {
            "model": SPARK_MODEL, "input": [{"type": "message", "role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
            "max_output_tokens": max_output_tokens, "stream": False, "store": False,
        })
        text = output_text(result)
        try:
            parsed = json.loads(text)
            validate_json(parsed, schema)
        except (ValueError, TypeError, KeyError):
            raise MetaAPIError("Spark output failed local schema validation", code="schema_validation_failed",
                               response=result) from None
        return {"response": result, "text": text, "parsed": parsed,
                "source": sources[0] if image_path is not None else None, "sources": sources,
                "usage": _reported_usage(result.get("usage")), "latency_seconds": self.ledger[-1]["latency_seconds"]}

    def segment_image(self, image_path: str | Path, prompt: str, *, decode_masks: bool = True) -> dict[str, Any]:
        if not prompt.strip() or len(prompt) > 2048:
            raise ValueError("Use one concise concept or supported object prompt")
        if decode_masks:
            _sam_decoder()  # Dependency check happens before paid inference.
        image_url, source = _image_data(image_path)
        result = self._request("POST", "responses", {
            "model": SAM_MODEL, "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": image_url},
            ]}], "stream": False, "store": False,
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        })
        text = output_text(result)
        try:
            objects = parse_sam_boxes(text)
        except ValueError:
            raise MetaAPIError("SAM output failed geometry parsing; inspect the local trace",
                               code="sam_parse_failed", response=result) from None
        if any(record["frame"] != 0 for record in objects):
            raise MetaAPIError("SAM image response contained a nonzero frame", code="sam_invalid_image_frame")
        if decode_masks:
            try:
                objects = decode_sam_masks(objects)
            except MetaAPIError as error:
                error.response = result
                raise
        return {"response": result, "text": text, "objects": objects, "source": source,
                "mask_decoding": "official_meta_sam_parser_0.1.0" if decode_masks else "skipped_explicitly",
                "usage": _reported_usage(result.get("usage")), "latency_seconds": self.ledger[-1]["latency_seconds"]}

    def usage_summary(self) -> dict[str, Any]:
        estimate = 0.0
        input_tokens = output_tokens = images = 0
        usage_unreported = 0
        unsuccessful_with_tokens = 0
        for entry in self.ledger:
            if entry.get("model") == SPARK_MODEL:
                if entry.get("status") != "ok" and "usage" not in entry:
                    continue  # No response usage survived an ambiguous failure.
                usage = _reported_usage(entry.get("usage"))
                if any(type(usage.get(field)) is not int or usage[field] < 0
                       for field in ("input_tokens", "output_tokens")):
                    usage_unreported += 1
                if entry.get("status") != "ok" and any(
                    type(usage.get(field)) is int and usage[field] >= 0
                    for field in ("input_tokens", "output_tokens")
                ):
                    unsuccessful_with_tokens += 1
                input_tokens += _reported_tokens(usage.get("input_tokens"))
                output_tokens += _reported_tokens(usage.get("output_tokens"))
            elif entry.get("model") == SAM_MODEL and entry.get("status") == "ok":
                images += 1
        estimate += input_tokens * 1.25 / 1_000_000 + output_tokens * 4.25 / 1_000_000
        estimate += images * 0.0025
        return {"request_attempts": len(self.ledger), "input_tokens": input_tokens,
                "output_tokens": output_tokens, "sam_images": images,
                "spark_requests_with_unreported_token_usage": usage_unreported,
                "unsuccessful_spark_requests_with_reported_token_usage": unsuccessful_with_tokens,
                "estimated_usd": round(estimate, 6),
                "estimate_note": "Published Standard rates; includes reported Spark tokens from rejected responses. "
                                 "Excludes unreported usage, unsuccessful SAM images and ambiguous charges. Not a billing hard stop.",
                "requests": self.ledger}
