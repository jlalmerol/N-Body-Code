#!/usr/bin/env python3
"""Summarize one N-body Tracy capture using only the Python standard library."""

import argparse
import ast
import csv
import json
import re
import statistics
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


def number(value):
    """Reject missing values and the SDK's UINT64_MAX duration sentinel."""
    try:
        result = Decimal(str(value).strip())
        if not result.is_finite() or abs(result) >= 2**64 - 1:
            return None
        return int(result) if result == result.to_integral_value() else float(result)
    except (InvalidOperation, ValueError, TypeError):
        return None


def read_csv(path, skip=0):
    with path.open(newline="") as stream:
        for _ in range(skip):
            next(stream, None)
        reader = csv.DictReader(stream, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        return [{str(k).strip(): v for k, v in row.items() if k is not None} for row in reader]


def write_csv(path, rows, fields):
    # Always emit a header, including when optional data is absent.
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    return dict(samples=len(values), mean=statistics.mean(values),
                median=statistics.median(values), min=min(values), max=max(values),
                stddev=statistics.stdev(values) if len(values) > 1 else "")


def attributes(value):
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value or "{}")
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError, TypeError):
            pass
    return {}


def summarize(run_dir, output_dir=None, skip_warmup=1, particles=None):
    run_dir = Path(run_dir).resolve()
    if run_dir.is_file():
        report = run_dir
        report_dir = report.parent
        root = report_dir.parent.parent if report_dir.parent.name == "reports" else report_dir
    else:
        reports = sorted(run_dir.rglob("ops_perf_results*.csv"))
        if len(reports) != 1:
            raise ValueError(f"Expected one operation report under {run_dir}, found {len(reports)}; pass a specific CSV")
        report = reports[0]
        report_dir = report.parent
        root = run_dir
        if root == report_dir and report_dir.parent.name == "reports":
            root = report_dir.parent.parent
    output_dir = Path(output_dir).resolve() if output_dir else report_dir / "summary"
    if skip_warmup < 0:
        raise ValueError("skip_warmup must be nonnegative")
    if particles is not None and particles <= 0:
        raise ValueError("particles must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    issues = []

    def warn(message):
        issues.append({"message": message})

    def optional(name):
        for candidate in (report_dir / name, root / ".logs" / name, root / name):
            if candidate.is_file():
                return candidate
        return None

    raw = read_csv(report)
    raw = [r for r in raw if (r.get("OP CODE") or "").startswith("NBody")]
    if not raw:
        raise ValueError(f"No N-body operation rows in {report}")
    raw.sort(key=lambda r: number(r.get("HOST START TS")) or 0)
    ids = [(r.get("DEVICE ID"), r.get("GLOBAL CALL COUNT")) for r in raw]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate device/operation IDs; repeated invocations may have been grouped")
    log = optional("run.log")
    log_text = log.read_text(errors="replace") if log else ""
    process_match = re.search(r"Using N\s*=\s*(\d+),\s*Ncycle\s*=\s*(\d+)", log_text)
    table = re.findall(r"^\s*(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+(\d+)\s*$", log_text, re.M)
    rank_counts = {int(row[1]) for row in table}
    single_rank = rank_counts == {1}
    if not single_rank:
        warn("Single-rank execution is unverified; interaction throughput is left blank")
    if process_match and single_rank and len(raw) != (int(process_match[2]) + 1) * len({r.get("DEVICE ID") for r in raw}):
        warn("Operation count differs from initialization plus Ncycle; capture may be incomplete")

    event_path = optional("profile_log_device.csv")
    frequency = None
    architecture = next((r.get("DEVICE ARCH") for r in raw if r.get("DEVICE ARCH")), "")
    if not architecture:
        match = re.search(r"Creating TopologyDiscovery for architecture:\s*(\w+)", log_text)
        architecture = match[1] if match else ""
    if event_path:
        with event_path.open() as stream:
            header = stream.readline()
        if not architecture:
            match = re.search(r"ARCH:\s*(\w+)", header)
            architecture = match[1] if match else ""
        match = re.search(r"CHIP_FREQ\[MHz\]:\s*([\d.]+)", header)
        frequency = number(match[1]) if match else None
        if not frequency:
            warn("Device log has no positive clock frequency; device-zone nanoseconds remain blank")
    by_device = defaultdict(list)
    for row in raw:
        by_device[row.get("DEVICE ID", "")].append(row)
    selected = set()
    operations = []
    measurements = []

    def metric(op, name, value, unit, source):
        if value is not None:
            measurements.append({"device_id": op["device_id"], "operation_id": op["operation_id"],
                                 "operation": op["operation"], "metric": name, "value": value,
                                 "unit": unit, "source": source})

    for device_id, rows in by_device.items():
        for index, row in enumerate(rows):
            op_id = row["GLOBAL CALL COUNT"]
            kept = index >= skip_warmup
            n = particles or number(attributes(row.get("ATTRIBUTES")).get("particles"))
            if n is None and process_match:
                n = int(process_match[1])
            cores = number(row.get("CORE COUNT"))
            duration = number(row.get("DEVICE KERNEL DURATION [ns]"))
            if duration is None or duration <= 0:
                duration = None
                missing = str(row.get("DEVICE KERNEL DURATION [ns]", "")).strip().lower() in {"", "nan", "none", "n/a"}
                warn(f"{'Missing' if missing else 'Invalid'} kernel duration for device {device_id}, operation {op_id}")
            op = {"device_id": device_id, "operation_id": op_id, "invocation": index,
                  "included": kept, "operation": row["OP CODE"], "particles": n,
                  "cores": cores, "math_fidelity": row.get("MATH FIDELITY", ""),
                  "kernel_ns": duration, "kernel_cycles": None,
                  "interactions_per_second": None, "interactions_per_second_per_core": None}
            start, end = (number(row.get(f"DEVICE KERNEL {part} CYCLE")) for part in ("START", "END"))
            if start is not None and end is not None and end >= start:
                op["kernel_cycles"] = end - start
            if single_rank and len(by_device) == 1 and n and duration:
                # Useful directed pairs; excludes padded work and self interactions.
                op["interactions_per_second"] = n * (n - 1) * 1e9 / duration
                if cores and cores > 0:
                    op["interactions_per_second_per_core"] = op["interactions_per_second"] / cores
            operations.append(op)
            if not kept:
                continue
            selected.add((device_id, op_id))
            for field, value in row.items():
                # Preserve actual reported counters and stage timings, not tensor shapes or timestamps.
                if field.endswith("[ns]") and ("DURATION" in field or "LATENCY" in field):
                    metric(op, field, duration if field == "DEVICE KERNEL DURATION [ns]" else number(value), "ns", "operation_report")
                elif "%" in field or "IPC" in field or "Issue Rate" in field:
                    metric(op, field, number(value), "%" if "%" in field else "ratio", "operation_report")
            metric(op, "kernel_cycles", op["kernel_cycles"], "cycles", "operation_report")
            metric(op, "useful_directed_interactions_per_second", op["interactions_per_second"], "interactions/s", "derived")
            metric(op, "useful_directed_interactions_per_second_per_core", op["interactions_per_second_per_core"], "interactions/s/core", "derived")
    if not selected:
        warn("No operations remain after warmup exclusion")
    configurations = defaultdict(set)
    for op in operations:
        configurations[(op['device_id'], op['operation'])].add((op['particles'], op['cores'], op['math_fidelity']))
    if any(len(settings) > 1 for settings in configurations.values()):
        raise ValueError("Particle count, core count or fidelity changes within one operation group; summarize separate runs")
    if not any("%" in m["metric"] or m["unit"] == "ratio" for m in measurements):
        warn("No hardware utilization/stall counters were recorded; stage duration includes waiting")

    # Preserve host scope samples and associate them using the enclosing force-call interval.
    host_rows = []
    host_path = optional("tracy_ops_times.csv")
    if host_path:
        scopes = read_csv(host_path)
        force_scopes = [s for s in scopes if s.get("name") == "NBody force calculation"]
        for scope in scopes:
            if scope.get("name") not in {"NBody force calculation", "NBody H2D", "NBody workload", "NBody D2H"}:
                continue
            start, duration = number(scope.get("ns_since_start")), number(scope.get("exec_time_ns"))
            if start is None or duration is None or duration < 0:
                continue
            parents = [s for s in force_scopes if s.get("thread") == scope.get("thread")
                       and number(s.get("ns_since_start")) is not None
                       and number(s.get("exec_time_ns")) is not None
                       and number(s["ns_since_start"]) <= start
                       and start + duration <= number(s["ns_since_start"]) + number(s["exec_time_ns"])]
            parent = min(parents, key=lambda s: number(s["exec_time_ns"])) if parents else None
            candidates = []
            if parent:
                lower = number(parent["ns_since_start"])
                upper = lower + number(parent["exec_time_ns"])
                candidates = [r for r in raw if number(r.get("HOST START TS")) is not None
                              and lower <= number(r["HOST START TS"]) <= upper]
            mapped = candidates[0] if len(candidates) == 1 else None
            key = (mapped.get("DEVICE ID", ""), mapped["GLOBAL CALL COUNT"]) if mapped else None
            kept = key in selected if mapped else False
            sample = {"device_id": key[0] if key else "", "operation_id": key[1] if key else "",
                      "thread": scope.get("thread", ""), "scope": scope["name"],
                      "start_ns": start, "duration_ns": duration, "included": kept}
            host_rows.append(sample)
            if kept:
                op = next(o for o in operations if (o["device_id"], o["operation_id"]) == key)
                metric(op, scope["name"], duration, "ns", "host_scope")
        if any(not s["operation_id"] for s in host_rows):
            warn("Some host scopes could not be mapped uniquely to an operation; excluded from statistics")
    else:
        warn("tracy_ops_times.csv is absent; no host scope summaries")

    # Pair device endpoints by core, RISC, invocation, zone and source location.
    zones = []
    if event_path:
        events = read_csv(event_path, skip=1)
        stacks = defaultdict(list)
        events.sort(key=lambda r: number(r.get("time[cycles since reset]")) or 0)
        for event in events:
            kind = event.get("type")
            if kind not in {"ZONE_START", "ZONE_END"}:
                continue
            clock = number(event.get("time[cycles since reset]"))
            if clock is None:
                continue
            key_fields = ("PCIe slot", "core_x", "core_y", "RISC processor type", "run host ID",
                          "trace id", "trace id counter", "zone name", "source line", "source file")
            key = tuple(event.get(k, "") for k in key_fields)
            if kind == "ZONE_START":
                stacks[key].append(clock)
                continue
            if not stacks[key]:
                warn(f"Unmatched device ZONE_END: {key}")
                continue
            start = stacks[key].pop()
            if clock < start:
                warn(f"Negative device zone duration: {key}")
                continue
            device_id, op_id = key[0], key[4]
            included = (device_id, op_id) in selected
            zones.append({"device_id": device_id, "operation_id": op_id,
                          "core_x": key[1], "core_y": key[2], "risc": key[3], "zone": key[7],
                          "source_line": key[8], "source_file": key[9],
                          "trace_id": key[5], "trace_counter": key[6],
                          "start_cycle": start, "duration_cycles": clock - start,
                          "duration_ns": (clock - start) * 1000 / frequency if frequency and frequency > 0 else None,
                          "included": included})
        unmatched = sum(len(s) for s in stacks.values())
        if unmatched:
            warn(f"{unmatched} unmatched device ZONE_START events")
        if zones and not any(z["included"] for z in zones):
            warn("Device event IDs could not be matched to selected operations")
        missing = selected - {(z['device_id'], z['operation_id']) for z in zones if z['included']}
        if missing:
            warn(f"No paired device zones for selected device/operation IDs: {sorted(missing)}")
    else:
        warn("profile_log_device.csv is absent; no per-core device zone summaries")

    # App EVAL includes host preparation and transfers. Preserve every sample,
    # but only summarize cycles whose force operation survived warmup exclusion.
    app_rows = []
    for row in table:
        n, ranks, prediction, evaluation, correction, cycle = row
        included = int(cycle) + 1 >= skip_warmup
        app_rows.append(dict(cycle=int(cycle), particles=int(n), ranks=int(ranks), included=included,
                             prediction_s=float(prediction), evaluation_s=float(evaluation), correction_s=float(correction)))
        if single_rank and len(by_device) == 1:
            matches = [o for o in operations if o['invocation'] == int(cycle) + 1]
            if included and len(matches) == 1:
                for name, value in (("app_prediction", prediction), ("app_EVAL", evaluation), ("app_correction", correction)):
                    metric(matches[0], name, float(value), "s", "app_log")

    summaries = []
    grouped = defaultdict(list)
    for item in measurements:
        grouped[(item["device_id"], item["operation"], item["metric"], item["unit"], item["source"])].append(item["value"])
    for (device_id, operation, name, unit, source), values in sorted(grouped.items()):
        summaries.append(dict(device_id=device_id, operation=operation, metric=name, unit=unit, source=source, **describe(values)))
    zone_summary = []
    grouped = defaultdict(list)
    for z in zones:
        if z["included"]:
            grouped[(z["device_id"], z["risc"], z["zone"], z["source_file"], z["source_line"])].append(z)
    for (device_id, risc, zone, source_file, source_line), samples in sorted(grouped.items()):
        stats = describe([s["duration_cycles"] for s in samples])
        zone_summary.append(dict(device_id=device_id, risc=risc, zone=zone, source_file=source_file,
                                 source_line=source_line, cores=len({(s["core_x"], s["core_y"]) for s in samples}),
                                 mean_ns=statistics.mean([s["duration_ns"] for s in samples]) if all(s["duration_ns"] is not None for s in samples) else "",
                                 **stats))
    core_summary = []
    grouped = defaultdict(list)
    for z in zones:
        if z["included"]:
            grouped[(z["device_id"], z["core_x"], z["core_y"], z["risc"], z["zone"],
                     z["source_file"], z["source_line"])].append(z["duration_cycles"])
    for (device_id, core_x, core_y, risc, zone, source_file, source_line), values in sorted(grouped.items()):
        core_summary.append(dict(device_id=device_id, core_x=core_x, core_y=core_y, risc=risc,
                                 zone=zone, source_file=source_file, source_line=source_line, **describe(values)))

    # App phase timers describe only the last call, unlike the host scope samples.
    last_phases = []
    for payload in re.findall(r"Last force phases \(s\):([^\n]+)", log_text):
        for name, value in re.findall(r"(\w+)=([\d.eE+-]+)", payload):
            last_phases.append({"phase": name, "seconds": float(value), "scope": "last_force_call_only"})
    manifest = optional("profile_manifest.json")
    metadata = [{"key": "architecture", "value": architecture},
                {"key": "device_clock_mhz_from_log", "value": frequency},
                {"key": "source_report", "value": str(report)},
                {"key": "skip_warmup_per_device", "value": skip_warmup},
                {"key": "operation_rows", "value": len(operations)},
                {"key": "included_operation_rows", "value": len(selected)}]
    if manifest:
        for key, value in json.loads(manifest.read_text()).items():
            metadata.append({"key": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value})
    else:
        warn("No profile_manifest.json; build flags and revision are unverified")

    write_csv(output_dir / "operations.csv", operations, list(operations[0]))
    write_csv(output_dir / "measurements.csv", measurements, ["device_id", "operation_id", "operation", "metric", "value", "unit", "source"])
    write_csv(output_dir / "summary.csv", summaries, ["device_id", "operation", "metric", "unit", "source", "samples", "mean", "median", "min", "max", "stddev"])
    write_csv(output_dir / "host_phases.csv", host_rows, ["device_id", "operation_id", "thread", "scope", "start_ns", "duration_ns", "included"])
    write_csv(output_dir / "device_zones.csv", zones, ["device_id", "operation_id", "core_x", "core_y", "risc", "zone", "source_line", "source_file", "trace_id", "trace_counter", "start_cycle", "duration_cycles", "duration_ns", "included"])
    write_csv(output_dir / "device_zone_summary.csv", zone_summary, ["device_id", "risc", "zone", "source_file", "source_line", "cores", "samples", "mean", "median", "min", "max", "stddev", "mean_ns"])
    write_csv(output_dir / "core_summary.csv", core_summary, ["device_id", "core_x", "core_y", "risc", "zone", "source_file", "source_line", "samples", "mean", "median", "min", "max", "stddev"])
    write_csv(output_dir / "app_cycles.csv", app_rows, ["cycle", "particles", "ranks", "included", "prediction_s", "evaluation_s", "correction_s"])
    write_csv(output_dir / "last_force_phases.csv", last_phases, ["phase", "seconds", "scope"])
    write_csv(output_dir / "metadata.csv", metadata, ["key", "value"])
    write_csv(output_dir / "diagnostics.csv", issues, ["message"])
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="one capture directory or its ops_perf_results CSV")
    parser.add_argument("--output-dir", type=Path, help="default: summary/ beside operation CSV")
    parser.add_argument("--skip-warmup", type=int, default=1, help="initial operations excluded per device (default: 1)")
    parser.add_argument("--particles", type=int, help="N, only needed when absent from metadata/log")
    args = parser.parse_args()
    try:
        output = summarize(args.run, args.output_dir, args.skip_warmup, args.particles)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Profile summary failed: {error}\n")
    print(f"Profile CSV summaries: {output}")


if __name__ == "__main__":
    main()
