from __future__ import annotations

import unittest

from resources.runtime_python_utils import is_probably_python_executable


class RuntimePythonUtilsTests(unittest.TestCase):
    def test_python_like_names_are_accepted(self) -> None:
        self.assertTrue(is_probably_python_executable("python.exe"))
        self.assertTrue(is_probably_python_executable("python3"))
        self.assertTrue(is_probably_python_executable("pythonw.exe"))

    def test_packaged_app_name_is_not_treated_as_python(self) -> None:
        self.assertFalse(is_probably_python_executable("NPEC Labeling Tool.exe"))
        self.assertFalse(is_probably_python_executable("my-gui-app.exe"))


if __name__ == "__main__":
    unittest.main()
