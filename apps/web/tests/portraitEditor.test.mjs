import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

const source = await readFile(new URL('../src/portraitEditorState.ts', import.meta.url), 'utf8')
const { code } = await transformWithOxc(source, 'portraitEditorState.ts', { target: 'es2022' })
const { changedEdge, drawnBounds, editedBounds, movedBounds, portraitPayload, portraitsChanged, portraitSourceMatches } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)

const original = {
  project_id: 'project-1', production_id: 'production-1', build_id: 'build-1',
  characters: [
    { character_id: 'kotone', name: '琴音', image_artifact_id: 'portrait-1', image_url: '/images/portrait-1', framing: 'upper_body', height_cm: 156, body_bounds: null },
    { character_id: 'guide', name: '案内役', image_artifact_id: 'portrait-2', image_url: '/images/portrait-2', framing: 'full_body', height_cm: null, body_bounds: { left: 0.1, top: 0.1, right: 0.9, bottom: 0.9 } },
  ],
}

test('drawing works in either direction, outside the image, and at its bottom corner', () => {
  assert.deepEqual(drawnBounds({ x: 0.8, y: 0.9 }, { x: 0.2, y: 0.3 }), { left: 0.2, top: 0.3, right: 0.8, bottom: 0.9 })
  assert.deepEqual(drawnBounds({ x: -0.2, y: -0.3 }, { x: 1.2, y: 1.3 }), { left: 0, top: 0, right: 1, bottom: 1 })
  assert.deepEqual(drawnBounds({ x: 1, y: 1 }, { x: 1, y: 1 }), { left: 0.95, top: 0.95, right: 1, bottom: 1 })
})

test('moving a selection preserves its dimensions when stopped by an image edge', () => {
  const bounds = { left: 0.2, top: 0.1, right: 0.7, bottom: 0.8 }
  assert.deepEqual(movedBounds(bounds, 0.6, -0.6), { left: 0.5, top: 0, right: 1, bottom: 0.7 })
  assert.deepEqual(movedBounds(bounds, -0.01, 0.05), { left: 0.19, top: 0.15, right: 0.69, bottom: 0.85 })
  assert.deepEqual(bounds, { left: 0.2, top: 0.1, right: 0.7, bottom: 0.8 })
})

test('numeric edges cannot reverse the selection or reduce it below five percent', () => {
  const bounds = { left: 0.2, top: 0.1, right: 0.7, bottom: 0.8 }
  assert.equal(changedEdge(bounds, 'left', 0.9).left, 0.65)
  assert.equal(changedEdge(bounds, 'bottom', 0).bottom, 0.15)
  assert.equal(changedEdge(bounds, 'right', 5).right, 1)
  assert.equal(changedEdge(bounds, 'top', -5).top, 0)
  assert.equal(changedEdge(bounds, 'top', NaN), bounds)
})

test('saving a body selection preserves every original height, framing and source image', () => {
  const before = structuredClone(original)
  const bounds = { left: 0.32, top: 0.23, right: 0.76, bottom: 0.94 }
  const edits = { kotone: bounds }
  const payload = portraitPayload(original, edits)
  assert.equal(payload.expected_build_id, 'build-1')
  assert.deepEqual(payload.characters[0], { character_id: 'kotone', framing: 'upper_body', height_cm: 156, image_artifact_id: 'portrait-1', body_bounds: bounds })
  assert.deepEqual(payload.characters[1], { character_id: 'guide', framing: 'full_body', height_cm: null, image_artifact_id: 'portrait-2', body_bounds: before.characters[1].body_bounds })
  assert.deepEqual(original, before)
  assert.equal(portraitsChanged(original, {}), false)
  assert.equal(portraitsChanged(original, edits), true)
  assert.equal(editedBounds(original.characters[0], edits), bounds)
})

test('an explicit automatic reset is sent as null and does not mutate the saved source', () => {
  const payload = portraitPayload(original, { guide: null })
  assert.equal(payload.characters[1].body_bounds, null)
  assert.equal(portraitsChanged(original, { guide: null }), true)
  assert.equal(portraitsChanged(original, { kotone: null }), false)
  assert.notEqual(original.characters[1].body_bounds, null)
})

test('late responses from a different project, production or build cannot become the editing source', () => {
  assert.equal(portraitSourceMatches(original, 'project-1', 'production-1', 'build-1'), true)
  assert.equal(portraitSourceMatches(original, 'project-2', 'production-1', 'build-1'), false)
  assert.equal(portraitSourceMatches(original, 'project-1', 'production-2', 'build-1'), false)
  assert.equal(portraitSourceMatches(original, 'project-1', 'production-1', 'build-2'), false)
})
