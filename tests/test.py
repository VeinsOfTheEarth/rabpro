import filecmp
import os
import time
import unittest

import rabpro as rp

# Run from top level repo directory
class DataTestCase(unittest.TestCase):
    def datatest(self, coords, da, force_merit, test_name, results_name):
        # Generate data
        rpo = rp.profiler(coords, name=test_name, da=da, force_merit=force_merit)
        rpo.delineate_basins()
        rpo.elev_profile()
        rpo.export("all")

        # Check if output files are equal
        test_elv_output = os.path.join("results", test_name, "dem_flowpath.json")
        check_elv_output = os.path.join("tests", "results", results_name, "dem_flowpath.json")
        self.assertTrue(filecmp.cmp(test_elv_output, check_elv_output, shallow=False))

        test_subbasin_output = os.path.join("results", test_name, "subbasins.json")
        check_subbasin_output = os.path.join("tests", "results", results_name, "subbasins.json")
        self.assertTrue(filecmp.cmp(test_subbasin_output, check_subbasin_output, shallow=False))

        # Check statistics
        statlist = ["min", "max", "range", "std", "sum", "pct50", "pct3"]
        data = rp.subbasin_stats.Dataset(
            "JRC/GSW1_3/GlobalSurfaceWater", "occurrence", stats=statlist
        )
        data, task = rpo.basin_stats([data], folder="rabpro test", test=True)

        stats = {
            "DA": 1993.9169921875,
            "count": 77323,
            "max": 100,
            "mean": 80.91875678233812,
            "min": 0,
            "p3": 22.384780056625935,
            "p50": 91.7284926534382,
            "range": 100,
            "sum": 4091273.6039215205,
        }
        print(stats, data["features"][0]["properties"])
        self.assertTrue(stats == data["features"][0]["properties"])

        for _ in range(12):
            if task.status()["state"] == "COMPLETED":
                break
            time.sleep(10)

        self.assertTrue(task.status()["state"] == "COMPLETED")


class BasicTest(DataTestCase):
    def test_files(self):
        coords = (56.22659, -130.87974)
        da = 1994
        self.datatest(coords, da, True, "basic_test_check", "basic_test")
