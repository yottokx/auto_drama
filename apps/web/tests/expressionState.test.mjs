import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

const source = await readFile(new URL('../src/expressionState.ts', import.meta.url), 'utf8')
const { code } = await transformWithOxc(source, 'expressionState.ts', { target: 'es2022' })
const { EXPRESSIONS, EXPRESSION_SETS, expressionName, expressionPicture, expressionsToDraw, faceBoxFrom, usedExpressions } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)

const picture = (outfitId, expression, stale = false) => ({ outfitId, expression, stale, imageArtifactId: `${outfitId}-${expression}`, baseImageArtifactId: 'base', imageUrl: `/api/artifacts/${outfitId}-${expression}/content` })

test('the fixed expressions are offered once each, and every set only adds to the one before', () => {
  const ids = EXPRESSIONS.map(item => item.id)
  assert.equal(ids.length, 11)
  assert.equal(new Set(ids).size, 11)
  // The ordinary face is the portrait itself, so it is never offered as a picture to make.
  assert.equal(ids.includes('neutral'), false)
  assert.deepEqual(EXPRESSION_SETS.map(set => set.expressions.length), [3, 7, 11])
  for (let index = 1; index < EXPRESSION_SETS.length; index += 1) {
    assert.deepEqual(EXPRESSION_SETS[index].expressions.slice(0, EXPRESSION_SETS[index - 1].expressions.length), EXPRESSION_SETS[index - 1].expressions)
  }
  assert.equal(expressionName('smile'), '微笑み')
  assert.equal(expressionName('unknown'), 'unknown')
})

test('a request draws only what a look lacks or has from a replaced portrait', () => {
  const pictures = [picture('default', 'smile'), picture('default', 'angry', true), picture('look1', 'smile')]
  assert.equal(expressionPicture(pictures, 'look1', 'smile').imageArtifactId, 'look1-smile')
  assert.equal(expressionPicture(pictures, 'look1', 'angry'), undefined)
  assert.deepEqual(expressionsToDraw(pictures, ['default'], ['smile', 'angry', 'sad']), [
    { outfitId: 'default', expression: 'angry' }, { outfitId: 'default', expression: 'sad' }])
  // Asked for every look, each look gets what it is missing.
  assert.deepEqual(expressionsToDraw(pictures, ['default', 'look1'], ['smile', 'angry']), [
    { outfitId: 'default', expression: 'angry' }, { outfitId: 'look1', expression: 'angry' }])
  assert.deepEqual(expressionsToDraw(pictures, [], ['smile']), [])
  assert.deepEqual(expressionsToDraw(pictures, ['default'], ['smile']), [])
})

test('the expressions a work uses are offered most used first', () => {
  assert.deepEqual(usedExpressions({ sad: 2, smile: 5, cry: 2, angry: 0 }), ['smile', 'sad', 'cry'])
  assert.deepEqual(usedExpressions(undefined), [])
})

test('a face is marked by a drag in any direction, kept inside the portrait, and not by a slip', () => {
  assert.deepEqual(faceBoxFrom({ x: 0.6, y: 0.3 }, { x: 0.4, y: 0.1 }), [0.4, 0.1, 0.6, 0.3])
  assert.deepEqual(faceBoxFrom({ x: 0.9, y: 0.05 }, { x: 1.4, y: -0.2 }), [0.9, 0, 1, 0.05])
  assert.equal(faceBoxFrom({ x: 0.5, y: 0.5 }, { x: 0.51, y: 0.8 }), null)
})
