"""
Upstash QStash Client for Durable Background Jobs, Cron Schedules, and Automated Retries.
"""

from __future__ import annotations

import hmac
import hashlib
import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger("orbit.qstash")


class QStashClient:
    """Client for Upstash QStash v2 REST API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        signing_key: Optional[str] = None,
    ):
        self.base_url = (base_url or os.getenv("QSTASH_URL", "https://qstash-us-east-1.upstash.io/v2")).rstrip("/")
        self.token = token or os.getenv("QSTASH_TOKEN", "")
        self.signing_key = signing_key or os.getenv("QSTASH_CURRENT_SIGNING_KEY", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.base_url)

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        if extra:
            headers.update(extra)
        return headers

    def publish_message(
        self,
        destination_url: str,
        body: Dict[str, Any],
        *,
        delay_seconds: Optional[int] = None,
        retries: int = 3,
        callback_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Publishes a durable background message to be dispatched to destination_url.
        Supports delays and automated retry backoffs.
        """
        if not self.is_configured:
            return {"error": "QStash is not configured in the environment."}

        url = f"{self.base_url}/publish/{destination_url}"
        custom_headers = {}
        if delay_seconds and delay_seconds > 0:
            custom_headers["Upstash-Delay"] = f"{delay_seconds}s"
        if retries is not None:
            custom_headers["Upstash-Retries"] = str(retries)
        if callback_url:
            custom_headers["Upstash-Callback"] = callback_url
        if headers:
            for k, v in headers.items():
                custom_headers[f"Upstash-Forward-{k}"] = v

        req = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(custom_headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                logger.info(f"[QStash] Dispatched message to {destination_url}: messageId={data.get('messageId')}")
                return data
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="replace")
            logger.error(f"[QStash] Publish failed ({e.code}): {err_msg}")
            return {"error": f"QStash HTTP {e.code}: {err_msg}"}
        except Exception as e:
            logger.error(f"[QStash] Publish request failed: {e}")
            return {"error": str(e)}

    def create_schedule(
        self,
        cron_expression: str,
        destination_url: str,
        body: Dict[str, Any],
        *,
        retries: int = 3,
    ) -> Dict[str, Any]:
        """Creates or registers a recurring cron schedule with QStash."""
        if not self.is_configured:
            return {"error": "QStash is not configured in the environment."}

        url = f"{self.base_url}/schedules/{destination_url}"
        custom_headers = {
            "Upstash-Cron": cron_expression,
            "Upstash-Retries": str(retries),
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(custom_headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                logger.info(f"[QStash] Created schedule '{cron_expression}' for {destination_url}: scheduleId={data.get('scheduleId')}")
                return data
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="replace")
            logger.error(f"[QStash] Create schedule failed ({e.code}): {err_msg}")
            return {"error": f"QStash HTTP {e.code}: {err_msg}"}
        except Exception as e:
            logger.error(f"[QStash] Create schedule request failed: {e}")
            return {"error": str(e)}

    def list_schedules(self) -> List[Dict[str, Any]]:
        """Lists all active schedules registered in QStash."""
        if not self.is_configured:
            return []
        url = f"{self.base_url}/schedules"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"[QStash] List schedules failed: {e}")
            return []

    def delete_schedule(self, schedule_id: str) -> bool:
        """Deletes a schedule by ID in QStash."""
        if not self.is_configured:
            return False
        url = f"{self.base_url}/schedules/{schedule_id}"
        req = urllib.request.Request(url, headers=self._headers(), method="DELETE")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status in (200, 204)
        except Exception as e:
            logger.error(f"[QStash] Delete schedule '{schedule_id}' failed: {e}")
            return False


_qstash_instance: Optional[QStashClient] = None


def get_qstash_client() -> QStashClient:
    global _qstash_instance
    if _qstash_instance is None:
        _qstash_instance = QStashClient()
    return _qstash_instance
