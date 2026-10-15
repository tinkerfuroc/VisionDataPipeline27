/** The server's 401 detail when a request has no valid session (server/vdp/api/gate.py LOGIN_REQUIRED). */
export const LOGIN_REQUIRED = "login required";
/** The close code of a WebSocket the login gate turned away (gate.py WS_LOGIN_REQUIRED). */
export const WS_LOGIN_REQUIRED = 4401;

/** Navigation behind an object so tests can watch it (jsdom does not implement location.assign). */
export const browser = {
  go(url: string): void {
    window.location.assign(url);
  },
};

/** Off to the login page, which returns here once logged in. */
export function toLogin(): void {
  browser.go(`/login?next=${encodeURIComponent(window.location.pathname + window.location.search)}`);
}

export const isLoginRequired = (status: number, body: unknown): boolean =>
  status === 401 &&
  body !== null &&
  typeof body === "object" &&
  (body as { detail?: unknown }).detail === LOGIN_REQUIRED;
