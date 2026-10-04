export function adjustmentFixture() {
  const positions = { left: { left: 100, top: 60, width: 300, height: 700 }, center: { left: 380, top: 60, width: 300, height: 700 }, right: { left: 660, top: 60, width: 300, height: 700 } }
  const characters = ['main', 'guide', 'later', 'extra'].map(character_id => ({ character_id, image_candidate_id: `${character_id}-image`, voice_candidate_id: `${character_id}-voice`, framing: 'upper_body', height_cm: 170, body_bounds: null, offset_y: 0, scale: 1 }))
  return {
    project_id: 'story', complete: true, readonly: false, busy: false,
    draft: { id: 'draft-1', revision: 1, status: 'draft', error: null, base_edition_id: 'edition-1', characters,
      geometry: { stage: { width: 960, height: 640 }, background: { fit: 'cover', position: 'center center', z_index: 0 }, slots: { left: 200, center: 480, right: 760 },
        message_window: { left: 20, top: 440, width: 920, height: 180, padding: { left: 24, top: 18, right: 24, bottom: 18 }, font_size: 24, line_spacing: 6, color: '#f3f4f6', background: '#101923', opacity: 230 / 255, z_index: 100 },
        characters: Object.fromEntries(characters.map(character => [character.character_id, { positions: structuredClone(positions) }])),
      },
    },
    cast: characters.map((character, index) => ({ character_id: character.character_id, name: ['主人公', '案内役', '後の章の人物', '比較用の人物'][index], role: index ? 'supporting' : 'main', chapter_numbers: index > 1 ? [2, 3] : [1, 2], image_candidate_id: character.image_candidate_id, voice_candidate_id: character.voice_candidate_id, image_url: `/image/${character.character_id}.png`, voice_url: `/voice/${character.character_id}.wav`, reference_text: '元の自己紹介です。', source_prompts: { image: '元の服装', voice: '元の声' }, result: { appearance: '元の服装', voice: '元の声', selfIntroduction: '元の自己紹介台詞です。', appliedInstructions: [] } })),
    scenes: [{ chapter_number: 1, scene_id: 'scene-a', title: '港の出会い', background_url: '/background/harbor.png', character_ids: ['main', 'guide'] }, { chapter_number: 2, scene_id: 'scene-b', title: '街の再会', background_url: '/background/town.png', character_ids: ['later'] }],
    candidates: characters.flatMap(character => ['image', 'voice'].map(kind => ({ id: `${character.character_id}-${kind}`, character_id: character.character_id, kind, artifact_id: `artifact-${character.character_id}-${kind}`, url: `/${kind}/${character.character_id}.${kind === 'image' ? 'png' : 'wav'}`, reference_text: kind === 'voice' ? '元の自己紹介です。' : null, source: 'original', prompt: null, job_id: null, sample_url: null }))),
    jobs: [], edition: { id: 'edition-1' }, limits: { upload_bytes: 33554432, image_max_side: 4096, audio_min_seconds: 0.25, audio_max_seconds: 30 },
  }
}

export function adjustmentMusicFixture() {
  const source = adjustmentFixture()
  source.scenes = source.scenes.map((scene, index) => ({ ...scene, production_id: `chapter-${index + 1}`, scene_id: 'scene-1', music_prompt: null }))
  source.draft.scene_music = source.scenes.map(scene => ({ production_id: scene.production_id, scene_id: scene.scene_id, candidate_id: null, action: 'stop', volume: .35 }))
  source.music_candidates = [{ id: 'music-1', production_id: 'chapter-1', scene_id: 'scene-1', source: 'generated', artifact_id: 'music-artifact-1', music_url: '/music/prepared.mp3', source_url: '/music/original.mp3', prompt: 'Genre: Quirky Pop. Instruments: Synthesizer, guitar.', loop_start_seconds: 8, loop_end_seconds: 72, duration_seconds: 72, source_duration_seconds: 120, job_id: null, status: 'completed', error: null, quality: { needs_review: false, near_silence_seconds: 0 } }]
  source.limits.music_upload_bytes = 33554432
  source.limits.music_max_seconds = 380
  return source
}

export function adjustmentContinuityFixture() {
  const source = adjustmentMusicFixture()
  const first = source.scenes[0]
  const transition = { visual: 'fade', duration_ms: 500, music_fade_out_ms: 1000, music_fade_in_ms: 1000 }
  source.scenes = ['港の出会い', '港での会話', '出航', '夜の街', '帰港'].map((title, index) => ({ ...first, scene_id: `scene-${index + 1}`, title }))
  source.scenes.push({ ...first, chapter_number: 2, production_id: 'chapter-2', scene_id: 'scene-1', title: '翌日の港' })
  source.music_candidates[0].scene_id = 'scene-1'
  source.music_candidates.push({ ...source.music_candidates[0], id: 'music-4', scene_id: 'scene-4', music_url: '/music/night.mp3', artifact_id: 'night-artifact' })
  source.draft.scene_music = source.scenes.map((scene, index) => {
    const action = ['play', 'continue', 'continue', 'stop', 'continue', 'stop'][index]
    const reason = `自動判断：${scene.title}`
    const cueTransition = action === 'continue' ? { ...transition, visual: 'dissolve', music_fade_out_ms: 0, music_fade_in_ms: 0 } : transition
    const row = { production_id: scene.production_id, scene_id: scene.scene_id, candidate_id: action === 'play' ? 'music-1' : null, action, volume: index === 0 ? .22 : .35, transition: structuredClone(cueTransition), reason }
    scene.music_plan = { action, candidate_id: row.candidate_id, source_scene_id: index < 3 ? 'scene-1' : null, reason, transition: structuredClone(cueTransition), prompt: 'A scene score.' }
    return row
  })
  // Default fixture is fully valid; tests introduce invalid continuation explicitly.
  source.draft.scene_music[4].action = 'stop'
  source.scenes[4].music_plan.action = 'stop'
  return source
}
