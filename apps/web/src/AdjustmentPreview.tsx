export type PreviewSlot = 'left' | 'center' | 'right'
export type PreviewPosition = { left: number; top: number; width: number; height: number }
export type AdjustmentGeometry = {
  stage: { width: number; height: number }
  background: { fit: 'cover'; position: string; z_index: number }
  message_window: {
    left: number; top: number; width: number; height: number
    padding: { left: number; top: number; right: number; bottom: number }
    font_size: number; line_spacing: number; color: string; background: string; opacity: number; z_index: number
  }
  slots: Record<PreviewSlot, number>
  characters: Record<string, { positions: Record<PreviewSlot, PreviewPosition> }>
}

/** Coordinates come from the same export calculation as the published VN. */
export function AdjustmentPreview({ geometry, backgroundUrl, characters, pending = false }: {
  geometry: AdjustmentGeometry; backgroundUrl: string | null
  characters: { id: string; name: string; imageUrl: string; slot: PreviewSlot }[]; pending?: boolean
}) {
  const { stage, message_window: message } = geometry
  return <div className="adjustment-stage-wrap" aria-busy={pending}>
    <svg className="adjustment-stage" viewBox={`0 0 ${stage.width} ${stage.height}`} role="img" aria-label="本編と同じ比率の表示プレビュー。人物とメッセージウインドウの位置を確認できます。">
      <rect width={stage.width} height={stage.height} fill="#25302d"/>
      {backgroundUrl ? <image href={backgroundUrl} width={stage.width} height={stage.height} preserveAspectRatio="xMidYMid slice"/> : <text x={stage.width / 2} y={100} textAnchor="middle" fill="#dce3db" fontSize={20}>この場面には背景画像がありません</text>}
      {characters.map(character => {
        const position = geometry.characters[character.id]?.positions[character.slot]
        return position ? <image key={character.id} href={character.imageUrl} x={position.left} y={position.top} width={position.width} height={position.height} preserveAspectRatio="none"><title>{character.name}</title></image> : null
      })}
      <rect x={message.left} y={message.top} width={message.width} height={message.height} fill={message.background} fillOpacity={message.opacity}/>
      <foreignObject x={message.left} y={message.top} width={message.width} height={message.height}>
        <div style={{ boxSizing: 'border-box', width: '100%', height: '100%', padding: `${message.padding.top}px ${message.padding.right}px ${message.padding.bottom}px ${message.padding.left}px`, color: message.color, fontFamily: 'sans-serif', fontSize: message.font_size, lineHeight: `${message.font_size + message.line_spacing}px`, overflow: 'hidden' }}>
          <div>{characters[0]?.name ?? '名前'}</div>
          <div>これは表示を確認するための台詞です。<br/>立ち絵の大きさや、ウインドウとの重なりを確認できます。</div>
        </div>
      </foreignObject>
    </svg>
    {pending && <span className="adjustment-preview-status" role="status">プレビューを更新中…</span>}
  </div>
}
