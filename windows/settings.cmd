@rem ============================================================
@rem  IOPaint launcher settings - edit this file, not the .bat's
@rem ============================================================

@rem Erase model: lama (best quality/speed), migan (fastest), mat, manga, cv2
set "IOPAINT_MODEL=lama"

@rem Web UI port
set "IOPAINT_PORT=8080"

@rem Background removal model:
@rem   isnet-general-use  - great for real photos, esp. on light backgrounds
@rem   briaai/RMBG-1.4    - strong all-rounder
@rem   birefnet-general   - highest quality, slowest, needs more RAM
@rem   u2net / silueta    - fastest, lower quality
set "IOPAINT_REMOVE_BG_MODEL=isnet-general-use"

@rem Alpha matting for the cutout edges: none | fast | quality
set "IOPAINT_ALPHA_MATTING=fast"

@rem Longest side used by the 'quality' matting solve (RAM/speed tradeoff)
set "IOPAINT_MATTING_MAX_SIZE=1280"

@rem CPU threads, 0 = auto detect physical cores (8 on a Ryzen 7 5700G)
set "IOPAINT_CPU_THREADS=0"

@rem Where models are stored (several GB). Change it to move them to another disk.
set "IOPAINT_MODEL_DIR=%USERPROFILE%\.cache\iopaint-models"

@rem Folder opened in the built-in file manager / used for auto saving.
@rem Leave empty to disable.
set "IOPAINT_OUTPUT_DIR=%USERPROFILE%\Pictures\IOPaint"

@rem What to do when the output file already exists:
@rem   unique    - add a counter, never overwrite (recommended)
@rem   timestamp - append date+time
@rem   overwrite - replace the old file
set "IOPAINT_SAVE_POLICY=unique"

@rem Open the browser automatically (true/false)
set "IOPAINT_INBROWSER=true"
