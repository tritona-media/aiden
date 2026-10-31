"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";

type IntroductionResponse = {
  message: string;
  has_cv: boolean;
};

async function responseError(response: Response): Promise<string> {
  const data: unknown = await response.json().catch(() => null);
  if (
    typeof data === "object" &&
    data !== null &&
    "detail" in data &&
    typeof data.detail === "string"
  ) {
    return data.detail;
  }
  return "Aiden ist gerade nicht erreichbar. Bitte versuche es später erneut.";
}

async function readIntroductionStream(
  response: Response,
  onMeta: (hasCv: boolean) => void,
  onToken: (text: string) => void,
): Promise<void> {
  if (!response.body) {
    throw new Error("Der Streaming-Response enthält keinen Inhalt.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let isComplete = false;

  function processLine(line: string) {
    if (!line.trim()) {
      return;
    }

    const event: unknown = JSON.parse(line);
    if (typeof event !== "object" || event === null || !("type" in event)) {
      throw new Error("Aiden hat ein ungültiges Streaming-Event gesendet.");
    }

    if (event.type === "meta" && "has_cv" in event && typeof event.has_cv === "boolean") {
      onMeta(event.has_cv);
    } else if (event.type === "token" && "text" in event && typeof event.text === "string") {
      onToken(event.text);
    } else if (event.type === "done") {
      isComplete = true;
    } else if (
      event.type === "error" &&
      "message" in event &&
      typeof event.message === "string"
    ) {
      throw new Error(event.message);
    } else {
      throw new Error("Aiden hat ein ungültiges Streaming-Event gesendet.");
    }
  }

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      processLine(line);
    }
    if (done) {
      if (buffer) {
        processLine(buffer);
      }
      break;
    }
  }

  if (!isComplete) {
    throw new Error("Aidens Antwort wurde unerwartet unterbrochen.");
  }
}

function getApiUrl() {
  return (
    process.env.NEXT_PUBLIC_AIDEN_API_URL ?? "http://127.0.0.1:8000"
  ).replace(/\/$/, "");
}

export default function Home() {
  const [welcomeMessage, setWelcomeMessage] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [hasCv, setHasCv] = useState(false);
  const [hasLoadedUser, setHasLoadedUser] = useState(false);
  const [userId, setUserId] = useState("");
  const [error, setError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadWelcomeMessage() {
      try {
        const storedUserId = localStorage.getItem("aiden-user-id");
        const validUserId =
          storedUserId &&
          /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
            storedUserId,
          )
            ? storedUserId
            : null;
        const currentUserId = validUserId ?? crypto.randomUUID();
        if (!validUserId) {
          localStorage.setItem("aiden-user-id", currentUserId);
        }
        setUserId(currentUserId);

        const response = await fetch(
          `${getApiUrl()}/api/users/${currentUserId}/intro/stream`,
          { signal: controller.signal },
        );
        if (!response.ok) {
          throw new Error(await responseError(response));
        }
        setWelcomeMessage("");
        setIsStreaming(true);
        await readIntroductionStream(
          response,
          (hasCv) => {
            setHasCv(hasCv);
            setIsLoading(false);
          },
          (text) => setWelcomeMessage((current) => current + text),
        );
        setHasLoadedUser(true);
      } catch (requestError) {
        if (!controller.signal.aborted) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "Aiden ist gerade nicht erreichbar. Bitte versuche es später erneut.",
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoading(false);
          setIsStreaming(false);
        }
      }
    }

    void loadWelcomeMessage();
    return () => controller.abort();
  }, []);

  async function uploadCv(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || !userId) {
      return;
    }

    setError("");
    setWelcomeMessage("");
    setIsUploading(true);
    setIsStreaming(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `${getApiUrl()}/api/users/${userId}/cv/stream`,
        { method: "POST", body: formData },
      );
      if (!response.ok) {
        throw new Error(await responseError(response));
      }
      await readIntroductionStream(
        response,
        setHasCv,
        (text) => setWelcomeMessage((current) => current + text),
      );
    } catch (uploadError) {
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Der Lebenslauf konnte nicht hochgeladen werden.",
      );
    } finally {
      setIsUploading(false);
      setIsStreaming(false);
      event.target.value = "";
    }
  }

  return (
    <main className="app-shell">
      <section className="chat-panel" aria-label="Conversation with Aiden">
        <div className="conversation">
          <div className="welcome" aria-live="polite">
            {isLoading && <p role="status">Aiden wird geladen…</p>}
            {!isLoading && welcomeMessage && (
              <h1 className={isStreaming ? "streaming" : undefined}>
                {welcomeMessage}
              </h1>
            )}
            {isStreaming && !welcomeMessage && <p role="status">Aiden schreibt…</p>}
            {!isLoading && error && <p role="alert">{error}</p>}
            {!isLoading && hasLoadedUser && !hasCv && !error && (
              <>
                <input
                  ref={fileInput}
                  className="file-input"
                  type="file"
                  accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                  aria-label="Lebenslauf als PDF oder DOCX auswählen"
                  onChange={uploadCv}
                />
                <button
                  className="upload-button"
                  type="button"
                  onClick={() => fileInput.current?.click()}
                  disabled={isUploading}
                >
                  {isUploading
                    ? "Lebenslauf wird hochgeladen…"
                    : "Lebenslauf hochladen"}
                </button>
                <p className="upload-hint">PDF oder DOCX, maximal 10 MB</p>
              </>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
