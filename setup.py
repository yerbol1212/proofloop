"""Include a previously built frontend in release wheels, without running Node."""
from pathlib import Path
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithFrontend(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / "web" / "dist"
        destination = Path(self.build_lib) / "proofloop" / "_web"
        if destination.exists():
            shutil.rmtree(destination)
        if (source / "index.html").is_file():
            shutil.copytree(source, destination)
        else:
            self.announce("Frontend not built: library/replay commands are available; build web/ before making a workbench release.", level=2)


setup(cmdclass={"build_py": BuildWithFrontend})
