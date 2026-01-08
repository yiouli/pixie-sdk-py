"""Test Langfuse initialization without credentials (Pixie-only mode)."""

import pytest
from langfuse import Langfuse
from langfuse._client.constants import PIXIE_ONLY_MODE_PLACEHOLDER


@pytest.fixture
def no_langfuse_credentials(monkeypatch):
    """Remove Langfuse credentials from environment."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)


def test_client_initialization_without_credentials(no_langfuse_credentials):
    """Test that Langfuse client can be initialized without credentials."""
    # Should not raise an exception
    client = Langfuse()

    # Verify client is initialized
    assert client is not None
    assert hasattr(client, "_resources")
    assert client._resources is not None


def test_placeholder_keys_used(no_langfuse_credentials):
    """Test that placeholder keys are used when credentials are missing."""
    client = Langfuse()

    # Verify placeholder key is used
    if client._resources:
        assert client._resources.public_key == PIXIE_ONLY_MODE_PLACEHOLDER


def test_tracer_available_without_credentials(no_langfuse_credentials):
    """Test that OTel tracer is available even without credentials."""
    client = Langfuse()

    # Verify tracer is available (not NoOpTracer)
    assert hasattr(client, "_otel_tracer")
    assert client._otel_tracer is not None
    # Should have a proper tracer, not NoOpTracer
    assert type(client._otel_tracer).__name__ != "NoOpTracer"


def test_server_export_disabled_without_credentials(no_langfuse_credentials):
    """Test that server export is disabled when credentials are missing."""
    client = Langfuse()

    # Check the span processor
    if client._resources and hasattr(client._resources, "tracer_provider"):
        provider = client._resources.tracer_provider
        if provider and hasattr(provider, "_active_span_processor"):
            processor = provider._active_span_processor

            # Find LangfuseSpanProcessor
            from langfuse._client.span_processor import LangfuseSpanProcessor

            if hasattr(processor, "_span_processors"):
                # MultiSpanProcessor
                lf_processors = [
                    p
                    for p in processor._span_processors
                    if isinstance(p, LangfuseSpanProcessor)
                ]
                if lf_processors:
                    assert not lf_processors[0].server_export_enabled
            elif isinstance(processor, LangfuseSpanProcessor):
                assert not processor.server_export_enabled


def test_auth_check_without_credentials(no_langfuse_credentials):
    """Test that auth_check returns True in Pixie-only mode without making API calls."""
    client = Langfuse()

    # Should return True without making API calls (no network error)
    result = client.auth_check()
    assert result is True
