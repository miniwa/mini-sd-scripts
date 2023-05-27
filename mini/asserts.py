from typing import TypeVar

CastT = TypeVar("CastT")

class Asserts:
  @staticmethod
  def check(condition: bool, message: str | None = None):
    if not condition:
      raise AssertionError(message if message is not None else "Assertion failed.")

  @staticmethod
  def cast_to(value: any, _type: type[CastT]) -> CastT:
    Asserts.check(isinstance(value, _type), f"Value {value} is not of type {_type}.")
    return value
