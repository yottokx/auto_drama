import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import vm from 'node:vm'
import { transformWithOxc } from 'vite'
import { adjustmentContinuityFixture } from './adjustmentFixture.mjs'

const dataUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
async function moduleUrl(file, replacements = {}) {
  let source = await readFile(new URL(`../src/${file}`, import.meta.url), 'utf8')
  for (const [name, url] of Object.entries(replacements)) source = source.replaceAll(`'${name}'`, JSON.stringify(url))
  const { code } = await transformWithOxc(source, file, { target: 'es2022', jsx: { runtime: 'automatic' } })
  return dataUrl(code.replace('"react/jsx-runtime"', JSON.stringify(import.meta.resolve('react/jsx-runtime'))))
}
const hooksUrl = dataUrl(`
let slots=[],cursor=0; export const effects=[];
export function reset(){slots=[];cursor=0;effects.length=0}
export function beginRender(){cursor=0;effects.length=0}
export function useState(initial){const i=cursor++;if(!(i in slots))slots[i]=typeof initial==='function'?initial():initial;return[slots[i],next=>{slots[i]=typeof next==='function'?next(slots[i]):next}]}
export function useRef(initial){const i=cursor++;if(!(i in slots))slots[i]={current:initial};return slots[i]}
export function useEffect(callback,dependencies){effects.push({callback,dependencies})}
`)
const hooks = await import(hooksUrl)
const stateUrl = await moduleUrl('adjustmentState.ts'), state = await import(stateUrl)
const runtimeUrl = dataUrl('export const chapterMusicRuntime=()=>globalThis;')
const { AdjustmentMusicSeamPreview, musicSeamScript } = await import(await moduleUrl('AdjustmentMusicSeamPreview.tsx', { react: hooksUrl, './adjustmentState': stateUrl, './musicRuntime': runtimeUrl }))
const runtimeCode = await readFile(new URL('../../../packages/tyrano_export/music.js', import.meta.url), 'utf8')
const transitionsCode = await readFile(new URL('../../../packages/tyrano_export/transitions.js', import.meta.url), 'utf8')
const settle = async () => { for (let i=0;i<8;i++) await new Promise(resolve=>setImmediate(resolve)) }
function walk(node, predicate) {
  if (!node || typeof node !== 'object') return []
  if (Array.isArray(node)) return node.flatMap(child=>walk(child,predicate))
  return [...(predicate(node)?[node]:[]), ...walk(node.props?.children,predicate)]
}
function text(node) {
  if(typeof node==='string'||typeof node==='number')return String(node)
  if(Array.isArray(node))return node.map(text).join('')
  return node?.props?text(node.props.children):''
}
function rows() {
  const source=adjustmentContinuityFixture()
  return state.adjustmentMusicContinuity(state.receiveAdjustment(state.initialAdjustmentEditor,source,'adopt'))
}
function harness(t,current=rows()[1],previous=rows()[0]) {
  hooks.reset()
  const keys=['window','AudioContext','fetch','setTimeout','clearTimeout','AutoDramaMusic','AutoDramaTransitions']
  const saved=Object.fromEntries(keys.map(key=>[key,globalThis[key]]))
  const contexts=[],fetches=[],timers=new Map();let wall=0,timerId=0,stops=0
  globalThis.setTimeout=(fn,delay)=>{const id=++timerId;timers.set(id,{fn,due:wall+delay});return id}
  globalThis.clearTimeout=id=>timers.delete(id)
  globalThis.window=globalThis
  globalThis.AudioContext=class {
    currentTime=wall/1000;state='running';sampleRate=48000;destination={};gains=[];sources=[];closed=false
    constructor(){contexts.push(this)}
    createGain(){const gain={gain:{value:1,values:[],ramps:[],setValueAtTime(value,at){this.value=value;this.values.push({value,at})},linearRampToValueAtTime(value,at){this.ramps.push({value,at})},cancelScheduledValues(){}},connect(){},disconnect(){}};this.gains.push(gain);return gain}
    createBufferSource(){const source={connect(){},disconnect(){},stop(){this.stopped=true},start(...args){this.started=args}};this.sources.push(source);return source}
    async resume(){this.state='running'}
    async close(){this.closed=true;this.state='closed'}
    async decodeAudioData(){return{duration:72,sampleRate:48000}}
  }
  let fetcher=async()=>({ok:true,arrayBuffer:async()=>new ArrayBuffer(8)})
  globalThis.fetch=async(url,options)=>{fetches.push({url,options});return fetcher(url,options)}
  vm.runInThisContext(runtimeCode);vm.runInThisContext(transitionsCode)
  let props={previous,current,active:false,onStart:()=>{props.active=true},onStop:()=>{stops++;props.active=false}},prior=[]
  function render(){
    hooks.beginRender();const tree=AdjustmentMusicSeamPreview(props)
    prior=hooks.effects.map((effect,index)=>{
      const old=prior[index]
      if(old&&effect.dependencies.length===old.dependencies.length&&effect.dependencies.every((value,i)=>Object.is(value,old.dependencies[i])))return old
      old?.cleanup?.();return{dependencies:effect.dependencies,cleanup:effect.callback()}
    });return tree
  }
  function button(label){const node=walk(render(),node=>node.type==='button'&&text(node)===label)[0];assert.ok(node,`missing ${label}`);return node}
  let closed=false
  function close(){if(!closed){closed=true;prior.forEach(effect=>effect.cleanup?.())}}
  t.after(()=>{close();for(const[key,value]of Object.entries(saved)){if(value===undefined)delete globalThis[key];else globalThis[key]=value}})
  render()
  return{render,button,contexts,fetches,close,get stops(){return stops},handle(fn){fetcher=fn},update(next){props={...props,...next}},async advance(milliseconds){const target=wall+milliseconds;for(;;){await settle();const next=[...timers].filter(([,value])=>value.due<=target).sort((a,b)=>a[1].due-b[1].due)[0];if(!next)break;wall=next[1].due;contexts.forEach(ctx=>ctx.currentTime=wall/1000);timers.delete(next[0]);next[1].fn()}wall=target;contexts.forEach(ctx=>ctx.currentTime=wall/1000);await settle()}}
}

test('seam script uses the resolved continuing track, its authored volume and bounded manifest filenames',()=>{
  const list=rows(),prepared=musicSeamScript(list[1],list[2])
  assert.equal(prepared.previousPosition,68)
  assert.equal(prepared.script.music_cues[0].volume,.22)
  assert.equal(prepared.script.music_cues[1].action,'continue')
  assert.equal(prepared.script.music_cues[1].asset_id,undefined)
  assert.equal(prepared.urls.get('./data/bgm/before.mp3'),'/music/prepared.mp3')
  assert.deepEqual(prepared.script.assets,[{id:'before',kind:'music',filename:'before.mp3'}])
})

test('real exported runtime preserves one source, its loop clock and gains when the next scene continues',async t=>{
  const h=harness(t)
  h.button('場面のつなぎ目を試聴').props.onClick();await settle();h.render()
  const ctx=h.contexts[0]
  assert.deepEqual(ctx.sources[0].started,[0,68])
  assert.equal(ctx.sources[0].loopStart,8)
  assert.equal(ctx.sources[0].loopEnd,72)
  const gains=ctx.gains.map(node=>node.gain.values.length)
  await h.advance(4000);h.render()
  assert.equal(ctx.sources.length,1)
  assert.deepEqual(ctx.gains.map(node=>node.gain.values.length),gains)
  assert.ok(text(h.render()).includes('次の場面 · 6秒'))
  await h.advance(6000);h.render()
  assert.equal(ctx.closed,true)
  assert.equal(ctx.sources[0].stopped,true)
  assert.equal(h.stops,1)
})

test('real exported fades stop the old source only after fading and start a changed cue from its introduction',async t=>{
  const list=rows(),current={...list[3],setting:{...list[3].setting,action:'play',candidate_id:'music-4'},candidate:adjustmentContinuityFixture().music_candidates[1]}
  const h=harness(t,current,list[2])
  h.button('場面のつなぎ目を試聴').props.onClick();await settle();h.render()
  const ctx=h.contexts[0]
  await h.advance(4000)
  assert.equal(ctx.sources[0].stopped,undefined)
  assert.ok(ctx.gains[1].gain.ramps.some(ramp=>ramp.value===0&&ramp.at===5))
  await h.advance(1000)
  assert.equal(ctx.sources[0].stopped,true)
  assert.deepEqual(ctx.sources[1].started,[0,0])
  assert.ok(ctx.gains[1].gain.ramps.some(ramp=>ramp.value===1&&ramp.at===6))
  assert.deepEqual(h.fetches.map(row=>row.url),['/music/prepared.mp3','/music/night.mp3'])
  await h.advance(1000);h.render()
  h.button('つなぎ目の試聴を停止').props.onClick()
  assert.equal(ctx.closed,true)
})

test('stopping BGM uses the exported fade before silence and never creates a new source',async t=>{
  const list=rows(),h=harness(t,list[3],list[2])
  h.button('場面のつなぎ目を試聴').props.onClick();await settle();h.render()
  const ctx=h.contexts[0]
  await h.advance(4000);assert.equal(ctx.sources[0].stopped,undefined)
  await h.advance(1000);assert.equal(ctx.sources[0].stopped,true)
  assert.equal(ctx.sources.length,1)
})

test('stale scene changes abort pending loads and dispose context without starting obsolete audio',async t=>{
  const h=harness(t);let resolve
  h.handle(()=>new Promise(done=>{resolve=done}))
  h.button('場面のつなぎ目を試聴').props.onClick();await settle();h.render()
  h.update({current:rows()[3]});h.render()
  assert.equal(h.fetches[0].options.signal.aborted,true)
  assert.equal(h.contexts[0].closed,true)
  resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(8)});await settle()
  assert.equal(h.contexts[0].sources.length,0)
  assert.equal(h.stops,1)
})

test('invalid continuation prevents seam audition while decode failures release its external context',async t=>{
  const list=rows(),invalid={...list[1],error:'継続するBGMがありません。'}
  const h=harness(t,invalid,list[0])
  assert.equal(h.button('場面のつなぎ目を試聴').props.disabled,true)
  h.update({current:list[1]});h.render()
  h.handle(async()=>({ok:false}))
  h.button('場面のつなぎ目を試聴').props.onClick();await settle();h.render()
  assert.equal(h.contexts[0].closed,true)
  assert.ok(text(h.render()).includes('BGMを準備できませんでした'))
})
