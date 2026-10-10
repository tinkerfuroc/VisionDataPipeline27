from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .deps import State
from .views import clip_summary

router = APIRouter(prefix="/api/projects", tags=["projects"])


class NameBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ColorBody(BaseModel):
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")


@router.get("")
def list_projects(state: State) -> list[dict]:
    return [p.model_dump(mode="json") for p in state.store.list_projects()]


@router.post("", status_code=201)
def create_project(body: NameBody, state: State) -> dict:
    return state.store.create_project(body.name.strip()).model_dump(mode="json")


@router.get("/{project_id}")
def get_project(project_id: str, state: State) -> dict:
    return state.store.read_project(project_id).model_dump(mode="json")


@router.post("/{project_id}/classes", status_code=201)
def add_class(project_id: str, body: NameBody, state: State) -> dict:
    return state.store.add_class(project_id, body.name).model_dump(mode="json")


@router.patch("/{project_id}/classes/{class_id}")
def set_class_color(project_id: str, class_id: int, body: ColorBody, state: State) -> dict:
    return state.store.set_class_color(project_id, class_id, body.color).model_dump(mode="json")


@router.get("/{project_id}/clips")
def list_clips(project_id: str, state: State) -> list[dict]:
    state.store.read_project(project_id)
    return [clip_summary(state, c) for c in state.store.list_clips(project_id)]
