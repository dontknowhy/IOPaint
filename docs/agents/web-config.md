# web_config.py (the Gradio config editor)

`iopaint/web_config.py:save_config` builds the pydantic model from `locals()` and is wired to a long positional list of
Gradio components, so adding a config field means editing the signature, the `save_btn.click` input list, and
`default_configs` together; `web_config.py` itself is a standalone Gradio config editor behind
`iopaint start-web-config` that shares nothing with the `web_app/` frontend, which is the UI actually served at `/`.
