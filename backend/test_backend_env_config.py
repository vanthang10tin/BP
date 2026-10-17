import os
import tempfile
import unittest
from unittest.mock import patch

import app


class BackendEnvConfigTests(unittest.TestCase):
    def test_loopback_cvat_url_uses_localhost_host_routing(self):
        with patch.dict(os.environ, {"CVAT_URL": "http://127.0.0.1:8080", "CVAT_ACCESS_TOKEN": "test-token"}):
            self.assertEqual(app.get_cvat_config(), ("http://localhost:8080", "test-token"))

    def test_cvat_headers_use_bearer_for_raw_or_prefixed_tokens(self):
        for token in ("test-token", "Token test-token", "Bearer test-token"):
            with self.subTest(token=token):
                self.assertEqual(app.get_cvat_headers(token)["Authorization"], "Bearer test-token")

    def test_saving_gemini_key_updates_env_file_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(app, "CURRENT_DIR", directory), patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
                with open(os.path.join(directory, ".env"), "w", encoding="utf-8") as env_file:
                    env_file.write("OTHER_SETTING=preserved\nGEMINI_API_KEY=old-key\n")

                app.save_gemini_api_key(" new-key ")

                with open(os.path.join(directory, ".env"), "r", encoding="utf-8") as env_file:
                    saved = env_file.read()

                self.assertEqual(saved, "OTHER_SETTING=preserved\nGEMINI_API_KEY=new-key\n")
                self.assertEqual(os.environ["GEMINI_API_KEY"], "new-key")

    def test_rejecting_empty_or_multiline_key_does_not_write_env_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(app, "CURRENT_DIR", directory):
                env_path = os.path.join(directory, ".env")
                with open(env_path, "w", encoding="utf-8") as env_file:
                    env_file.write("OTHER_SETTING=preserved\n")

                for value in ("", "key\nINJECTED=value"):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        app.save_gemini_api_key(value)

                with open(env_path, "r", encoding="utf-8") as env_file:
                    self.assertEqual(env_file.read(), "OTHER_SETTING=preserved\n")

    def test_cvat_url_strips_api_suffix_and_normalizes(self):
        with patch.dict(os.environ, {"CVAT_URL": "https://app.cvat.ai/api/", "CVAT_ACCESS_TOKEN": "pat-123"}):
            self.assertEqual(app.get_cvat_config(), ("https://app.cvat.ai", "pat-123"))

    def test_request_headers_override_cvat_config(self):
        class DummyRequest:
            headers = {"x-cvat-url": "https://app.cvat.ai/api", "x-cvat-token": "custom-pat"}

        self.assertEqual(app.get_cvat_config(DummyRequest()), ("https://app.cvat.ai", "custom-pat"))

    def test_cvat_headers_include_user_agent_and_flexible_accept(self):
        headers = app.get_cvat_headers("pat-123")
        self.assertIn("application/json", headers["Accept"])
        self.assertIn("QC-Studio-2D", headers["User-Agent"])


if __name__ == "__main__":
    unittest.main()

