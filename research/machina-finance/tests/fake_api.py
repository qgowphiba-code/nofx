"""The fake API lives in the package (machina_finance.fake) so `run --dry-run` can use it too."""
from machina_finance.fake import BASE, FakeAPI  # noqa: F401
