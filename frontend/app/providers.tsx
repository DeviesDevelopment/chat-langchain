// app/providers.tsx
'use client'

import { ChakraProvider } from '@chakra-ui/react'
import theme from './theme' // adjust path if necessary

type Props = {
  children: React.ReactNode
}

export function Providers({ children }: Props) {
  return <ChakraProvider theme={theme}>{children}</ChakraProvider>
}
