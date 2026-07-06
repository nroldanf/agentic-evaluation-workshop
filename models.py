"""Pydantic schemas for extracting structured clinical data from a transcript.

These models are the target schema for structured output. LangChain/LangGraph
support this natively:

  - Agent:  `create_agent(model=..., response_format=ClinicalNote)`, then read
            the validated instance from `result["structured_response"]`.
  - Model:  `llm.with_structured_output(ClinicalNote)` for a bare chat model.

The Field descriptions double as extraction hints for the model, so keep them
clear and specific.
"""

from pydantic import BaseModel, Field


class VitalSigns(BaseModel):
    """Objective vital sign measurements recorded during the encounter.

    Each vital is split into a numeric/string value and its unit of measurement.
    """

    temperature: float | None = Field(
        default=None,
        description="Body temperature value, e.g. 100.8.",
    )
    temperature_units: str | None = Field(
        default=None,
        description="Unit for temperature, e.g. 'F' or 'C'.",
    )
    oxygen_saturation: float | None = Field(
        default=None,
        description="Peripheral oxygen saturation (SpO2) value, e.g. 95.",
    )
    oxygen_saturation_units: str | None = Field(
        default=None,
        description="Unit for oxygen saturation, typically '%'.",
    )
    heart_rate: int | None = Field(
        default=None,
        description="Heart rate value, e.g. 72.",
    )
    heart_rate_units: str | None = Field(
        default=None,
        description="Unit for heart rate, typically 'bpm'.",
    )
    respiratory_rate: int | None = Field(
        default=None,
        description="Respiratory rate value, e.g. 16.",
    )
    respiratory_rate_units: str | None = Field(
        default=None,
        description="Unit for respiratory rate, typically 'breaths/min'.",
    )
    blood_pressure: str | None = Field(
        default=None,
        description="Blood pressure value as 'systolic/diastolic', e.g. '120/80'.",
    )
    blood_pressure_units: str | None = Field(
        default=None,
        description="Unit for blood pressure, typically 'mmHg'.",
    )


class Diagnosis(BaseModel):
    """A single diagnosis with its ICD-10 code.

    Common model reused for both the differential diagnoses and the
    assessment's primary/working diagnosis.
    """

    icd10_code: str = Field(
        description="The ICD-10-CM code for the condition, e.g. 'J18.9'.",
    )
    diagnosis_name: str = Field(
        description="Human-readable name of the condition, e.g. 'Pneumonia, unspecified organism'.",
    )


class DifferentialDiagnosis(Diagnosis):
    """A possible condition considered during the evaluation."""


class Assessment(Diagnosis):
    """The clinician's primary or working diagnosis for the encounter."""


class DiagnosesOutput(BaseModel):
    """Diagnoses portion of a clinical note (differentials + assessment).

    Used as the focused output schema for the diagnoses node so the model isn't
    also responsible for vitals in the same pass. Merged with `VitalSigns` into a
    full `ClinicalNote`.
    """

    differential_diagnoses: list[DifferentialDiagnosis] = Field(
        default_factory=list,
        description="List of possible conditions considered based on the evaluation, including ICD-10 codes.",
    )
    assessment: Assessment = Field(
        description="Primary diagnosis or working diagnosis, including ICD-10 codes.",
    )


class ClinicalNote(BaseModel):
    """Structured clinical note extracted from a patient-doctor transcript."""

    vital_signs: VitalSigns = Field(
        default_factory=VitalSigns,
        description="Vital signs measured during the visit.",
    )
    differential_diagnoses: list[DifferentialDiagnosis] = Field(
        default_factory=list,
        description="List of possible conditions considered based on the evaluation, including ICD-10 codes.",
    )
    assessment: Assessment = Field(
        description="Primary diagnosis or working diagnosis, including ICD-10 codes.",
    )
