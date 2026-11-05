"""Protocol fixtures test analysis, not hardware performance."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('summary', Path(__file__).resolve().parents[1] / 'scripts/summarize_profile.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def write(path, rows):
    summary.write_csv(path, rows, list(rows[0]))


class ProfileSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report_dir = self.root / 'reports' / 'timestamp'
        self.report_dir.mkdir(parents=True)
        self.logs = self.root / '.logs'
        self.logs.mkdir()
        self.report = self.report_dir / 'ops_perf_results_fixture.csv'
        self.rows = []
        # The initial call is deliberately much slower than the measured calls.
        for i, duration in enumerate([100000, 1000, 2000, 3000]):
            self.rows.append({'OP CODE': 'NBodySFPU', 'DEVICE ID': '0',
                              'DEVICE ARCH': 'arbitrary_arch', 'GLOBAL CALL COUNT': str((i + 1) * 1024),
                              'HOST START TS': str(100 + i * 10000), 'CORE COUNT': '2',
                              'MATH FIDELITY': 'HiFi4', 'ATTRIBUTES': json.dumps({'particles': 32}),
                              'DEVICE KERNEL DURATION [ns]': str(duration),
                              'DEVICE KERNEL START CYCLE': str(2**54 + i * 10000),
                              'DEVICE KERNEL END CYCLE': str(2**54 + i * 10000 + duration),
                              'SFPU Avg (%)': '50'})
        write(self.report, self.rows)
        (self.root / 'run.log').write_text('Using N = 32, Ncycle = 3\n'
                                         '32 1 0.00001 0.002 0.00002 0\n'
                                         'Last force phases (s): pack=0.0001 H2D=0.0002\n')

    def load(self, name):
        return summary.read_csv(self.report_dir / 'summary' / name)

    def test_warmup_statistics_cycles_and_throughput(self):
        summary.summarize(self.root)
        operations = self.load('operations.csv')
        self.assertEqual([r['included'] for r in operations], ['False', 'True', 'True', 'True'])
        self.assertEqual(operations[1]['kernel_cycles'], '1000')  # no float timestamp rounding
        metrics = {r['metric']: r for r in self.load('summary.csv')}
        stats = metrics['DEVICE KERNEL DURATION [ns]']
        self.assertEqual(int(stats['samples']), 3)
        self.assertEqual(float(stats['mean']), 2000)
        self.assertEqual(float(stats['median']), 2000)
        self.assertEqual(float(stats['stddev']), 1000)
        self.assertEqual(float(metrics['SFPU Avg (%)']['mean']), 50)
        self.assertEqual(float(metrics['app_EVAL']['mean']), 0.002)
        self.assertEqual(float(operations[1]['interactions_per_second']), 32 * 31 * 1e6)
        self.assertEqual(self.load('last_force_phases.csv')[0]['scope'], 'last_force_call_only')

    def test_invalid_duration_and_unknown_rank_count_stay_blank(self):
        self.rows[1]['DEVICE KERNEL DURATION [ns]'] = str(2**64 - 1)
        write(self.report, self.rows)
        (self.root / 'run.log').unlink()
        summary.summarize(self.root)
        operations = self.load('operations.csv')
        self.assertEqual(operations[1]['kernel_ns'], '')
        self.assertTrue(all(r['interactions_per_second'] == '' for r in operations))
        self.assertTrue(any('Invalid kernel' in r['message'] for r in self.load('diagnostics.csv')))

    def test_host_scopes_map_by_force_envelope(self):
        scopes = []
        for i in range(4):
            scopes.extend([
                dict(name='NBody force calculation', ns_since_start=i * 10000, exec_time_ns=9000, thread=1),
                dict(name='NBody H2D', ns_since_start=i * 10000 + 10, exec_time_ns=20, thread=1)])
        write(self.logs / 'tracy_ops_times.csv', scopes)
        summary.summarize(self.root)
        host = self.load('host_phases.csv')
        h2d = [r for r in host if r['scope'] == 'NBody H2D']
        self.assertEqual([r['operation_id'] for r in h2d], ['1024', '2048', '3072', '4096'])
        stats = [r for r in self.load('summary.csv') if r['metric'] == 'NBody H2D'][0]
        self.assertEqual(stats['samples'], '3')
        self.assertEqual(float(stats['mean']), 20)

    def test_raw_zones_nested_pairing_zero_clock_and_unmatched(self):
        fields = ['PCIe slot', 'core_x', 'core_y', 'RISC processor type', 'run host ID',
                  'trace id', 'trace id counter', 'zone name', 'source line', 'source file',
                  'type', 'time[cycles since reset]']
        events = []
        for kind, clock in [('ZONE_START', 2**54), ('ZONE_START', 2**54 + 2),
                            ('ZONE_END', 2**54 + 5), ('ZONE_END', 2**54 + 10),
                            ('ZONE_END', 2**54 + 11)]:
            events.append(dict(zip(fields, ['0', '1', '1', 'TRISC_1', '2048', '', '',
                                            'compute', '10', 'kernel.cpp', kind, clock])))
        path = self.logs / 'profile_log_device.csv'
        with path.open('w', newline='') as f:
            f.write('ARCH: arbitrary_arch, CHIP_FREQ[MHz]: 0, Max Compute Cores: 2\n')
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(events)
        summary.summarize(self.root)
        zones = self.load('device_zones.csv')
        self.assertEqual([r['duration_cycles'] for r in zones], ['3', '10'])
        self.assertEqual(float(self.load('core_summary.csv')[0]['mean']), 6.5)
        self.assertTrue(all(r['duration_ns'] == '' for r in zones))
        self.assertTrue(any('Unmatched' in r['message'] for r in self.load('diagnostics.csv')))
        path.write_text(path.read_text().replace('CHIP_FREQ[MHz]: 0', 'CHIP_FREQ[MHz]: 500'))
        summary.summarize(self.root)
        self.assertEqual(float(self.load('device_zones.csv')[0]['duration_ns']), 6)

    def test_duplicate_ids_and_multiple_reports_rejected(self):
        self.rows[1]['GLOBAL CALL COUNT'] = '1024'
        write(self.report, self.rows)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summary.summarize(self.root)
        write(self.report_dir / 'ops_perf_results_second.csv', self.rows)
        with self.assertRaisesRegex(ValueError, 'Expected one'):
            summary.summarize(self.root)

    def test_specific_report_path_and_zero_remaining_samples(self):
        summary.summarize(self.report, skip_warmup=4)
        self.assertEqual(self.load('summary.csv'), [])
        self.assertTrue(any('No operations remain' in r['message'] for r in self.load('diagnostics.csv')))

    def test_zero_duration_is_not_summarized_and_manifest_is_preserved(self):
        self.rows[1]['DEVICE KERNEL DURATION [ns]'] = '0'
        write(self.report, self.rows)
        (self.root / 'profile_manifest.json').write_text(json.dumps({
            'sdk_revision': 'abc123', 'build_flags': {'NBODY_PHASE_TIMERS': 'ON'}}))
        summary.summarize(self.root)
        stats = [r for r in self.load('summary.csv') if r['metric'] == 'DEVICE KERNEL DURATION [ns]'][0]
        self.assertEqual(stats['samples'], '2')
        metadata = {r['key']: r['value'] for r in self.load('metadata.csv')}
        self.assertEqual(metadata['sdk_revision'], 'abc123')
        self.assertEqual(json.loads(metadata['build_flags'])['NBODY_PHASE_TIMERS'], 'ON')

    def test_host_only_architecture_metadata_and_missing_device_durations(self):
        for row in self.rows:
            row['DEVICE ARCH'] = ''
            row['DEVICE KERNEL DURATION [ns]'] = ''
            row['DEVICE KERNEL START CYCLE'] = ''
            row['DEVICE KERNEL END CYCLE'] = ''
        write(self.report, self.rows)
        with (self.root / 'run.log').open('a') as stream:
            stream.write('Creating TopologyDiscovery for architecture: example_arch (topology.cpp:1)\n')
        summary.summarize(self.root)
        metadata = {r['key']: r['value'] for r in self.load('metadata.csv')}
        self.assertEqual(metadata['architecture'], 'example_arch')
        self.assertTrue(all(not r['kernel_ns'] for r in self.load('operations.csv')))
        self.assertTrue(any('Missing kernel duration' in r['message'] for r in self.load('diagnostics.csv')))


if __name__ == '__main__':
    unittest.main()
