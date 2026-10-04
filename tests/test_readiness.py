"""Actual HTTP readiness is distinct from process liveness."""
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.mark.parametrize("config,manager,ready", [({},None,False),({"location":{}},None,False),({"location":{}},SimpleNamespace(is_loaded=False),False),({"location":{"name":"Test"}},SimpleNamespace(is_loaded=True),True)])
def test_readiness_checks_config_and_loaded_model(monkeypatch,config,manager,ready):
    monkeypatch.setattr(app.state,"config",config,raising=False)
    monkeypatch.setattr(app.state,"model_manager",manager,raising=False)
    import pandas as pd
    import numpy as np
    from src.data import preprocess
    monkeypatch.setattr(preprocess,"load_processed",lambda:pd.DataFrame({"date":pd.date_range("2026-01-01",periods=35),"temp_c":np.arange(35)+20.}))
    client=TestClient(app)
    response=client.get("/api/ready")
    assert response.status_code == (200 if ready else 503)
    assert response.json()["ready"] is ready
    assert response.json()["checks"] == {"config":bool(config.get("location",{}).get("name")),"models":bool(manager and manager.is_loaded),"data":True}
    assert client.get("/api/health").status_code == 200
    assert client.get("/health").json() == {"status":"ok"}

@pytest.mark.parametrize("config", [{"other":True},{"location":{}},{"location":{"name":""}},{"location":{"name":None}},{"location":{"name":"   "}}])
def test_missing_forecast_location_is_not_ready(monkeypatch,config):
    monkeypatch.setattr(app.state,"config",config,raising=False)
    monkeypatch.setattr(app.state,"model_manager",SimpleNamespace(is_loaded=True),raising=False)
    response=TestClient(app).get("/api/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["config"] is False

@pytest.mark.parametrize("kind", ["missing","corrupt","empty","columns","dates","infinite","too_short","valid"])
def test_required_inference_data_is_usable(monkeypatch,kind):
    import numpy as np
    import pandas as pd
    from src.data import preprocess
    df=pd.DataFrame({"date":pd.date_range("2026-01-01",periods=35),"temp_c":np.arange(35,dtype=float)+20})
    if kind=="empty":df=df.iloc[:0]
    if kind=="columns":df=df.drop(columns=["temp_c"])
    if kind=="dates":df["date"]="not-a-date"
    if kind=="infinite":df["temp_c"]=np.inf
    if kind=="too_short":df=df.iloc[:3]
    def load():
        if kind=="missing":raise FileNotFoundError("private/path")
        if kind=="corrupt":raise ValueError("private contents")
        return df
    monkeypatch.setattr(preprocess,"load_processed",load)
    monkeypatch.setattr(app.state,"config",{"location":{"name":"Test"}},raising=False)
    monkeypatch.setattr(app.state,"model_manager",SimpleNamespace(is_loaded=True),raising=False)
    response=TestClient(app).get("/api/ready")
    assert response.status_code == (200 if kind=="valid" else 503)
    assert response.json()["checks"]["data"] is (kind=="valid")
    assert "private" not in response.text

@pytest.mark.parametrize("kind",["missing","corrupt","empty","valid"])
def test_readiness_reads_actual_parquet_file(monkeypatch,tmp_path,kind):
    import numpy as np
    import pandas as pd
    from src.data import preprocess
    path=tmp_path/"daily_merged.parquet"
    if kind=="corrupt":path.write_text("not parquet")
    elif kind in ("empty","valid"):
        frame=pd.DataFrame({"date":pd.date_range("2026-01-01",periods=35),"temp_c":np.arange(35)+20.})
        (frame if kind=="valid" else frame.iloc[:0]).to_parquet(path)
    monkeypatch.setattr(preprocess,"_PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(preprocess,"_load_config",lambda:{"data":{"processed_dir":"."}})
    monkeypatch.setattr(app.state,"config",{"location":{"name":"Test"}},raising=False)
    monkeypatch.setattr(app.state,"model_manager",SimpleNamespace(is_loaded=True),raising=False)
    response=TestClient(app).get("/api/ready")
    assert response.status_code == (200 if kind=="valid" else 503)

@pytest.mark.parametrize("model,rows,ready",[("sarima",15,False),("sarima",16,False),("sarima",17,True),("lgbm",15,True),("tft",45,False),("tft",46,False),("tft",47,True),("tft-short",24,False),("tft-short",25,True)])
def test_engineered_history_covers_model_and_three_day_horizon(monkeypatch,model,rows,ready):
    import numpy as np
    import pandas as pd
    from src.data import preprocess
    manager=SimpleNamespace(is_loaded=True,sarima=None,tft=None,lgbm_models={})
    if model=="sarima":manager.sarima=SimpleNamespace(is_fitted=True)
    elif model.startswith("tft"):
        encoder=8 if model=="tft-short" else 30
        manager.tft=SimpleNamespace(is_fitted=True,config={"max_encoder_length":encoder},train_dataset=None)
    else:manager.lgbm_models={1:object()}
    monkeypatch.setattr(app.state,"config",{"location":{"name":"Test"}},raising=False)
    monkeypatch.setattr(app.state,"model_manager",manager,raising=False)
    monkeypatch.setattr(preprocess,"load_processed",lambda:pd.DataFrame({"date":pd.date_range("2026-01-01",periods=rows),"temp_c":np.arange(rows)+20.}))
    assert TestClient(app).get("/api/ready").status_code == (200 if ready else 503)

def test_duplicate_calendar_dates_are_not_ready(monkeypatch):
    import pandas as pd
    import numpy as np
    from src.data import preprocess
    df=pd.DataFrame({"date":pd.date_range("2026-01-01",periods=35),"temp_c":np.arange(35)+20.})
    df.loc[20,"date"]=df.loc[19,"date"]
    monkeypatch.setattr(preprocess,"load_processed",lambda:df)
    monkeypatch.setattr(app.state,"config",{"location":{"name":"Test"}},raising=False)
    monkeypatch.setattr(app.state,"model_manager",SimpleNamespace(is_loaded=True),raising=False)
    assert TestClient(app).get("/api/ready").status_code == 503
