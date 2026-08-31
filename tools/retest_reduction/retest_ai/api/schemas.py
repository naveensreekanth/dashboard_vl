from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional

class PreRetestEvent(BaseModel):
    """
    Strict input schema for a single ATE failure event.
    Only approved pre-retest observable features are accepted.
    Any outcome or leakage column will raise a validation error.
    """
    model_config = ConfigDict(extra="forbid")

    Device_ID: Optional[str] = Field(None, description="Device identifier (metadata only, not a model feature)")
    Failure_Event: Optional[int] = Field(None, description="Failure event index (metadata only)")
    Wafer_ID: str = Field(..., description="Wafer ID (e.g. W001, W002)")
    ATE_Site: int = Field(..., description="ATE Site / Socket (e.g. 1 to 10)")
    Fail_Test: str = Field(..., description="Test suite that failed (e.g. MBIST_03, Scan_145)")
    Fail_Bin: int = Field(..., description="Hardware failure bin (e.g. 45, 23, 12)")
    First_Result: str = Field("FAIL", description="Initial test result (FAIL)")
    Voltage_V: float = Field(..., description="Operating voltage during test in Volts")
    Temperature_C: float = Field(..., description="Die temperature in Celsius")
    First_Test_Time_sec: float = Field(..., description="Initial test execution time in seconds")
    Test_Month: int = Field(12, description="Test month index (0, 6, 12)")

class BatchEventItem(BaseModel):
    """
    Batch event item allowing optional Device_ID and Failure_Event metadata.
    Outcome leakage fields are strictly forbidden.
    """
    model_config = ConfigDict(extra="forbid")

    Device_ID: Optional[str] = Field(None, description="Device identifier (metadata only)")
    Failure_Event: Optional[int] = Field(None, description="Failure event index (metadata only)")
    Wafer_ID: str = Field(..., description="Wafer ID")
    ATE_Site: int = Field(..., description="ATE Site / Socket")
    Fail_Test: str = Field(..., description="Failing test name")
    Fail_Bin: int = Field(..., description="Failure bin")
    First_Result: str = Field("FAIL", description="First result")
    Voltage_V: float = Field(..., description="Voltage in V")
    Temperature_C: float = Field(..., description="Temperature in C")
    First_Test_Time_sec: float = Field(..., description="First test time in s")
    Test_Month: int = Field(12, description="Test month")

class BatchPredictionRequest(BaseModel):
    events: List[BatchEventItem]

class SinglePredictionResponse(BaseModel):
    Device_ID: Optional[str] = None
    Failure_Event: Optional[int] = None
    probability_retest_beneficial: float
    probability_percent: float
    probability_base: Optional[float] = None
    probability_adapted: Optional[float] = None
    online_adaptation_active: Optional[bool] = None
    recommendation: str
    policy_label: str
    policy_threshold: float
    estimated_retest_time_sec: Optional[float] = None
    predicted_retest_time_sec: Optional[float] = None
    model: str
    version: Optional[str] = None

class BatchPredictionItem(BaseModel):
    Device_ID: Optional[str] = None
    Failure_Event: Optional[int] = None
    probability_retest_beneficial: float
    probability_percent: Optional[float] = None
    probability_base: Optional[float] = None
    probability_adapted: Optional[float] = None
    online_adaptation_active: Optional[bool] = None
    recommendation: str
    policy_label: Optional[str] = None
    policy_threshold: Optional[float] = None
    estimated_retest_time_sec: Optional[float] = None
    predicted_retest_time_sec: Optional[float] = None
    model: Optional[str] = None
    version: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem]

class HealthResponse(BaseModel):
    status: str
    service: str

class ModelInfoResponse(BaseModel):
    model: str
    version: str
    target: str
    features: List[str]
    training_data: List[str]
    inference_data: str
    calibration_enabled: bool
    selection_reason: Optional[str] = None
    decision_policy_label: Optional[str] = None
    decision_policy_threshold: Optional[float] = None
