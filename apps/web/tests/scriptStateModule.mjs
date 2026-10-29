import { readFile } from 'node:fs/promises'
import { transformWithOxc } from 'vite'

const dataUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
async function transformed(file, replacements = {}) {
  let source = await readFile(new URL(`../src/${file}`, import.meta.url), 'utf8')
  for (const [name, url] of Object.entries(replacements)) source = source.replaceAll(`'${name}'`, JSON.stringify(url))
  const { code } = await transformWithOxc(source, file, { target: 'es2022' })
  return dataUrl(code)
}

/** The script tab's state module, importable from a data URL like the modules that use it. */
export const scriptStateUrl = await transformed('scriptTimelineState.ts', { './expressionState': await transformed('expressionState.ts') })
