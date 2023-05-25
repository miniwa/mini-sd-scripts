class Asserts:
  @staticmethod
  def check(condition: bool, message: str | None = None):
    if not condition:
      raise AssertionError(message if message is not None else "Assertion failed.")
