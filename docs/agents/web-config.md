# web_config.py (the Gradio config editor)

`iopaint/web_config.py:save_config` builds the pydantic model from `locals()` (minus the Gradio-injected
`request`, which is not one of the components) and is wired to a long positional list of Gradio components, so
adding a config field means editing the signature, the `save_btn.click` input list, and `default_configs`
together; `build_demo(config_file, i18n)` assembles the `gr.Blocks` and `main()` only launches it, so tests can
serve it on a free port. `web_config.py` itself is a standalone Gradio config editor behind
`iopaint start-web-config` that shares nothing with the `web_app/` frontend, which is the UI actually served at `/`.
