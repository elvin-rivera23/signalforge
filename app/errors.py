from fastapi import HTTPException, status

from app.schemas import ErrorCode, ErrorDetail, ErrorResponse


def http_error(
    code: ErrorCode, message: str, http_status=status.HTTP_400_BAD_REQUEST, hint: str | None = None
):
    detail = ErrorDetail(code=code, message=message, hint=hint)
    return HTTPException(status_code=http_status, detail=detail.model_dump())


def envelope_from_http_exception(exc: HTTPException) -> ErrorResponse:
    d = exc.detail
    if isinstance(d, dict) and "code" in d and "message" in d:
        return ErrorResponse(error=d)  # already our shape
    # Fallback to INTERNAL_ERROR envelope
    return ErrorResponse(
        error=ErrorDetail(code=ErrorCode.INTERNAL_ERROR, message=str(d), hint=None).model_dump()
    )
