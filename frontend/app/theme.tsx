// app/theme.tsx
import { extendTheme } from '@chakra-ui/react'

const theme = extendTheme({
  fonts: {
    heading: 'var(--font-fira-sans)',
    body: 'var(--font-fira-sans)',
  },
})

export default theme