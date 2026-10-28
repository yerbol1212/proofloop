import type { CSSProperties } from 'react'

export type IconName =
  | 'plus'
  | 'check'
  | 'close'
  | 'info'
  | 'camera'
  | 'image'
  | 'upload'
  | 'play'
  | 'wrench'
  | 'fork'
  | 'spoon'
  | 'chevron'
  | 'refresh'
export function Icon({
  name,
  size = 20,
  className,
  style,
}: {
  name: IconName
  size?: number
  className?: string
  style?: CSSProperties
}) {
  const paths: Record<IconName, React.ReactNode> = {
    plus: <path d="M12 5v14M5 12h14" />,
    check: <path d="m5 12 4 4L19 6" />,
    close: <path d="m6 6 12 12M18 6 6 18" />,
    info: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v6M12 7v.1" />
      </>
    ),
    camera: (
      <>
        <path d="M8 5 6.5 8H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-9a2 2 0 0 0-2-2h-2.5L16 5Z" />
        <circle cx="12" cy="14" r="4" />
      </>
    ),
    image: (
      <>
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <circle cx="8" cy="8" r="1.5" />
        <path d="m3 17 6-6 4 4 3-3 5 5" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5" />
      </>
    ),
    play: (
      <path d="m8 5 12 7-12 7Z" fill="currentColor" strokeLinejoin="round" />
    ),
    wrench: (
      <path d="M14 5a6 6 0 0 0-7 7L2 18a2.8 2.8 0 0 0 4 4l6-6a6 6 0 0 0 7-7l-4 4-4-4 4-4Z" />
    ),
    fork: (
      <>
        <path d="M7 2v7a5 5 0 0 0 10 0V2M12 2v20" />
      </>
    ),
    spoon: (
      <>
        <ellipse cx="12" cy="7" rx="5" ry="6" />
        <path d="M12 13v9" />
      </>
    ),
    chevron: <path d="m9 5 7 7-7 7" />,
    refresh: (
      <>
        <path d="M20 7v5h-5M4 17v-5h5" />
        <path d="M5.5 7a7.5 7.5 0 0 1 12-2L20 8M4 16l2.5 3a7.5 7.5 0 0 0 12-2" />
      </>
    ),
  }
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
      style={style}
    >
      {paths[name]}
    </svg>
  )
}
