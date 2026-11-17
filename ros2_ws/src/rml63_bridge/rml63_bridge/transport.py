"""Pluggable local output transport for observation packets.

The bridge never publishes ROS2 messages and never sends commands. It hands
each assembled Observation to a transport object:

  - "log": print to the ROS2 logger (default, no network, no sockets)
  - "zmq":  one-way ZMQ PUB socket, ROS2 bridge -> LeRobot only

The ZMQ transport is strictly one-directional: it has no receive method, no
action/command/target/trigger fields, and must never gain them. The ZMQ
socket is created in ``start()`` (never at module import time) and pyzmq is
imported lazily, so LogTransport and ObservationAssembler keep working
without pyzmq installed.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from .observation import Observation

DEFAULT_TRANSPORT = "log"
DEFAULT_ZMQ_ADDRESS = "tcp://127.0.0.1:5557"

# The exact, fixed key set of a serialized observation. Anything not listed
# here must NOT appear in the JSON (in particular no action / command /
# target_joint / trigger fields).
OBSERVATION_FIELDS = (
    "timestamp",
    "joint_position",
    "joint_velocity",
    "joint_current",
    "tcp_pose",
    "arm_status",
    "joint_enabled",
    "tool_state",
    "valid",
    "invalid_reasons",
)


def _jsonable(value: Any) -> Any:
    """Recursively convert a value to plain JSON-able Python types.

    - None stays None (serializes to JSON null);
    - numpy scalars (np.float32 etc.) -> float/int via .item();
    - numpy arrays -> nested lists via .tolist();
    - ROS2 message objects (get_fields_and_field_types) -> dict of fields;
    - dataclasses -> dict of fields;
    - anything else -> str() (never silently crash the sender).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item) and value.ndim == 0:
        # numpy scalar: np.float32(1.5).item() -> 1.5
        return _jsonable(value.item())
    if hasattr(value, "tolist") and callable(value.tolist):
        # numpy ndarray or array.array
        return _jsonable(value.tolist())
    if hasattr(value, "get_fields_and_field_types"):
        # ROS2 message object (e.g. Rmplusstate tool_state)
        return _jsonable({f: getattr(value, f) for f in value.get_fields_and_field_types()})
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable({k: getattr(value, k) for k in value.__dataclass_fields__})
    return str(value)


def observation_to_dict(obs: Observation) -> dict:
    """Convert an Observation to a plain-Python dict with the fixed key set.

    Missing data fields are None (JSON null) — never zero-filled.
    """
    return {name: _jsonable(getattr(obs, name)) for name in OBSERVATION_FIELDS}


def observation_to_json(obs: Observation) -> str:
    """Serialize an Observation to a JSON string (UTF-8)."""
    return json.dumps(observation_to_dict(obs))


def resolve_transport_name(param_value: str, env_value: Optional[str]) -> str:
    """Parameter wins over environment; empty/unset falls back to 'log'.

    Env vars: RML63_BRIDGE_TRANSPORT, RML63_BRIDGE_ZMQ_ADDRESS. When nothing
    is set, LogTransport is used (ZMQ is never enabled by default).
    """
    if param_value:
        return param_value
    if env_value:
        return env_value
    return DEFAULT_TRANSPORT


class ObservationTransport(ABC):
    """Interface every output sink must implement."""

    @abstractmethod
    def send(self, obs: Observation) -> None:
        """Deliver one observation frame."""


class LogTransport(ObservationTransport):
    """Development sink: prints the frame to the ROS2 logger (no ROS2 topic)."""

    def __init__(self, logger):
        self._logger = logger

    def send(self, obs: Observation) -> None:
        if obs.valid:
            self._logger.info(
                "obs valid | ts=%.6f | pos[0..2]=%s | vel[0..2]=%s | cur[0..2]=%s | tcp=%s | status=%s",
                obs.timestamp,
                obs.joint_position[:3] if obs.joint_position else None,
                obs.joint_velocity[:3] if obs.joint_velocity else None,
                obs.joint_current[:3] if obs.joint_current else None,
                obs.tcp_pose,
                obs.arm_status,
            )
        else:
            self._logger.warning("obs INVALID | reasons=%s", obs.invalid_reasons)


class ZmqObservationTransport(ObservationTransport):
    """One-way ZMQ PUB transport: ROS2 bridge -> LeRobot.

    - PUB socket, default address tcp://127.0.0.1:5557;
    - send() only publishes observations; there is deliberately NO receive
      method and no way to take commands back from the subscriber;
    - pyzmq is imported lazily inside start(), so this module (and the
      LogTransport / ObservationAssembler tests) work without pyzmq;
    - the socket is created in start(), never at import time;
    - close() is idempotent; send() after close() re-opens the socket
      instead of crashing (a closed PUB simply drops frames, it never
      raises into the ROS2 callback).
    """

    def __init__(self, address: str = DEFAULT_ZMQ_ADDRESS, context: Any = None):
        self._address = address
        self._context = context  # shared context allowed; None -> own context
        self._owns_context = context is None
        self._socket = None

    def start(self) -> "ZmqObservationTransport":
        """Create and bind the PUB socket. Safe to call repeatedly."""
        if self._socket is None:
            import zmq  # lazy: module import must work without pyzmq

            if self._context is None:
                self._context = zmq.Context()
            self._socket = self._context.socket(zmq.PUB)
            # Note: a PUB socket drops frames until a subscriber connects
            # (slow-joiner); LeRobot must connect before/at recording start.
            self._socket.bind(self._address)
        return self

    def send(self, obs: Observation) -> None:
        """Serialize and publish one observation. Never receives anything."""
        if self._socket is None:
            self.start()  # tolerate send-after-close / send-before-start
        self._socket.send_string(observation_to_json(obs))

    def close(self) -> None:
        """Idempotent close; must never crash on repeat calls."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        if self._owns_context and self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
            self._context = None


def make_transport(name: str, logger, address: Optional[str] = None):
    """Transport factory. Add new transports here later."""
    if name == "log":
        return LogTransport(logger)
    if name == "zmq":
        return ZmqObservationTransport(address or DEFAULT_ZMQ_ADDRESS)
    raise ValueError(f"unknown transport: {name!r} (supported: 'log', 'zmq')")
