import type { ApiErrorBody } from "./types";

export class ApiError extends Error {
  readonly code: string;
  readonly requestId: string | null;
  readonly status: number;

  constructor(status: number, body: ApiErrorBody) {
    super(body.error.message);
    this.name = "ApiError";
    this.code = body.error.code;
    this.requestId = body.error.request_id;
    this.status = status;
  }
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

async function parseJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!text) {
    return {} as T;
  }
  return JSON.parse(text) as T;
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  const isFormData = typeof FormData !== "undefined" && init?.body instanceof FormData;
  const headers: Record<string, string> = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...((init?.headers as Record<string, string> | undefined) ?? {}),
  };
  // FormData must not force application/json — browser sets multipart boundary.
  if (isFormData) {
    delete headers["Content-Type"];
  }
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    if (err instanceof Error && err.name === "AbortError") {
      throw err;
    }
    throw err;
  }

  if (!response.ok) {
    try {
      const body = await parseJson<ApiErrorBody>(response);
      if (body.error?.code) {
        throw new ApiError(response.status, body);
      }
    } catch (err) {
      if (err instanceof ApiError) {
        throw err;
      }
    }
    throw new Error(`Request failed with status ${response.status}`);
  }

  return parseJson<T>(response);
}
