from setuptools import setup

package_name = "rml63_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name, package_name + ".dora_nodes"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dys",
    maintainer_email="dys@local",
    description="Read-only RML63 ROS2 state bridge (observation only, no control)",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "rml63_bridge_node = rml63_bridge.bridge_node:main",
            "rml63_replay_observation = rml63_bridge.replay_observation:main",
        ],
    },
)
