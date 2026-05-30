/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MODEL_BASE_URL?: string
  // Optional default model id, used when no model is selected in the UI.
  readonly VITE_MODEL_NAME?: string
  readonly VITE_MODEL_API_KEY?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
