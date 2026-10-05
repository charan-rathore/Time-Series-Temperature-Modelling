"""Offline checks of the real fetch retry boundary."""
from unittest.mock import Mock
import pytest
import requests
from src.data.fetcher import _get_with_retry

@pytest.mark.parametrize("header",["9999999","garbage","-5"])
def test_retry_after_is_bounded_and_no_sleep_after_final_attempt(monkeypatch,header):
    response=Mock(status_code=429,headers={"Retry-After":header})
    response.raise_for_status.side_effect=requests.HTTPError(response=response)
    get=Mock(return_value=response);sleep=Mock()
    monkeypatch.setattr("src.data.fetcher.requests.get",get);monkeypatch.setattr("src.data.fetcher.time.sleep",sleep)
    with pytest.raises(RuntimeError):_get_with_retry("https://test.invalid",{},max_retries=2)
    assert get.call_count==2
    assert sleep.call_count==1
    assert 0 <= sleep.call_args[0][0] <= 60

def test_non_retryable_auth_failure_is_not_retried(monkeypatch):
    response=Mock(status_code=401);response.raise_for_status.side_effect=requests.HTTPError(response=response)
    get=Mock(return_value=response);sleep=Mock()
    monkeypatch.setattr("src.data.fetcher.requests.get",get);monkeypatch.setattr("src.data.fetcher.time.sleep",sleep)
    with pytest.raises(requests.HTTPError):_get_with_retry("https://test.invalid",{})
    assert get.call_count==1; sleep.assert_not_called()

def test_transport_failure_retries_and_can_recover(monkeypatch):
    response=Mock(status_code=200);response.json.return_value={"ok":True}
    get=Mock(side_effect=[requests.ConnectionError("offline"),response]);sleep=Mock()
    monkeypatch.setattr("src.data.fetcher.requests.get",get);monkeypatch.setattr("src.data.fetcher.time.sleep",sleep)
    assert _get_with_retry("https://test.invalid",{})=={"ok":True}
    assert get.call_count==2; assert sleep.call_count==1
