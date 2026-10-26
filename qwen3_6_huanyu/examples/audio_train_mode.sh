#!/usr/bin/env bash
# Source before native imports. Component flags take precedence over the legacy
# all-audio flag; missing component flags inherit its original behavior.
huanyu_resolve_audio_train_mode() {
    local legacy="${HUANYU_FREEZE_AUDIO-false}" default_train key value
    case "$legacy" in
        true) default_train=false ;;
        false) default_train=true ;;
        *) echo '[huanyu-audio-mode] FATAL: HUANYU_FREEZE_AUDIO must be true or false' >&2; return 2 ;;
    esac
    for key in HUANYU_TRAIN_AUDIO_ENCODER HUANYU_TRAIN_AUDIO_PROJECTOR; do
        value="${!key-$default_train}"
        case "$value" in
            true|false) export "$key=$value" ;;
            *) echo "[huanyu-audio-mode] FATAL: $key must be true or false" >&2; return 2 ;;
        esac
    done
    if [[ "$HUANYU_TRAIN_AUDIO_ENCODER" == true || "$HUANYU_TRAIN_AUDIO_PROJECTOR" == true ]]; then
        export HUANYU_FREEZE_AUDIO=false MINICPMO_TUNE_AUDIO=1
    else
        export HUANYU_FREEZE_AUDIO=true MINICPMO_TUNE_AUDIO=0
    fi
}
huanyu_resolve_audio_train_mode || return $?
