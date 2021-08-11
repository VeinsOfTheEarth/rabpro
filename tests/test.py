import filecmp
import os
import time
import unittest

import numpy as np
import rabpro as rp

# Run from top level repo directory
class DataTestCase(unittest.TestCase):
    def datatest(self, coords, da, force_merit, test_name, results_name, radius=None):
        print(f"Running {test_name}...")

        # Generate data
        self.rpo = rp.profiler(coords, name=test_name, da=da, force_merit=force_merit)
        self.rpo.delineate_basins(search_radius=radius)
        self.rpo.elev_profile()
        self.rpo.export("all")

        print(f"method: {self.rpo.method}")
        print(f"nrows: {self.rpo.nrows}")
        print(f"ncols: {self.rpo.ncols}")

        # Check if output files are equal
        test_elv_output = os.path.join("results", test_name, "dem_flowpath.json")
        check_elv_output = os.path.join(
            "tests", "results", results_name, "dem_flowpath.json"
        )
        self.assertTrue(filecmp.cmp(test_elv_output, check_elv_output, shallow=False))

        test_subbasin_output = os.path.join("results", test_name, "subbasins.json")
        check_subbasin_output = os.path.join(
            "tests", "results", results_name, "subbasins.json"
        )
        self.assertTrue(
            filecmp.cmp(test_subbasin_output, check_subbasin_output, shallow=False)
        )

    def metatest(self, method, nrows, ncols):
        self.assertEqual(self.rpo.method, method)
        self.assertEqual(self.rpo.nrows, nrows)
        self.assertEqual(self.rpo.ncols, ncols)

    def stattest(self, stats, datasets, length=1):
        # Check statistics
        data, task = self.rpo.basin_stats(datasets, folder="rabpro test", test=True)

        # Only check one set of stats for time-series data, but check length is equal
        self.assertEqual(len(data["features"]), length)

        ret_stats = data["features"][0]["properties"]
        print(f"Expected stats: {stats}")
        print(f"Returned stats: {ret_stats}")

        # Checking if the values are approximately equal due to GEE bug
        # yielding slightly diff values on identical runs
        self.assertEqual(stats.keys(), ret_stats.keys())
        np.testing.assert_allclose(
            np.array(list(stats.values())), np.array(list(ret_stats.values())), rtol=1e-03
        )

        for _ in range(12):
            status = task.status()["state"]
            if status in ["READY", "RUNING", "COMPLETED"]:
                self.assertTrue(True)
            time.sleep(10)

        self.assertTrue(task.status()["state"] in ["READY", "RUNING", "COMPLETED"])


class MERITTest(DataTestCase):
    def test_files(self):
        coords = (56.22659, -130.87974)
        da = 1994

        stats = {
            "DA": 1993.9169921875,
            "count": 77323,
            "max": 100,
            "mean": 80.91875678233811,
            "min": 0,
            "p3": 22.38478005662594,
            "p50": 91.7426459876037,
            "range": 100,
            "sum": 4091273.6039215205,
        }

        statlist = ["min", "max", "range", "sum", "pct50", "pct3"]
        data = rp.subbasin_stats.Dataset(
            "JRC/GSW1_3/GlobalSurfaceWater", "occurrence", stats=statlist
        )

        self.datatest(coords, da, True, "merit_test_check", "merit_test")
        self.metatest("merit", 50, 50)
        self.stattest(stats, [data])

    def test_imgcol(self):
        coords = (56.22659, -130.87974)
        da = 1994

        stats = {"DA": 1993.9169921875, "count": 4017116, "mean": 0}

        data = rp.subbasin_stats.Dataset(
            "JRC/GSW1_3/MonthlyHistory",
            "water",
            stats=["count"],
            start="2020-10-01",
            end="2020-12-10",
        )

        self.datatest(coords, da, True, "merit_imgcol_check", "merit_imgcol")
        self.metatest("merit", 50, 50)
        self.stattest(stats, [data], length=3)

    def test_radius(self):
        coords = (56.22659, -130.87974)
        da = 1994
        self.datatest(coords, da, True, "merit_radius_check", "merit_radius", radius=1000)
        self.metatest("merit", 22, 39)

    def test_shapefile(self):
        coords = os.path.join("tests", "data", "test_coords.shp")
        da = 1994
        self.datatest(coords, da, True, "merit_shapefile_check", "merit_shapefile")

    def test_csv(self):
        coords = os.path.join("tests", "data", "test_coords.csv")
        da = 1994
        self.datatest(coords, da, True, "merit_csv_check", "merit_csv")


class HydroBasinsTest(DataTestCase):
    def test_files(self):
        coords = (56.22659, -130.87974)
        da = 1994

        stats = {
            "DA": 2121.3,
            "count": 91173,
            "mean": 80.52335409462798,
            "min": 0,
            "p3": 22.328632410436462,
            "p50": 91.62934589830562,
            "stdDev": 32.377446695978065,
            "sum": 4757363.33725484,
        }

        statlist = ["min", "stdDev", "sum", "pct50", "pct3"]
        data = rp.subbasin_stats.Dataset(
            "JRC/GSW1_3/GlobalSurfaceWater", "occurrence", stats=statlist
        )

        self.datatest(coords, da, False, "hydro_test_check", "hydro_test")
        self.metatest("hydrobasins", 50, 50)
        self.stattest(stats, [data])


if __name__ == "__main__":
    unittest.main()
