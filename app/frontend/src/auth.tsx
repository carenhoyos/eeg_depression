import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { api, tokenStore } from './api/client'
import type { User } from './types'

interface AuthState { user: User | null; loading: boolean; login: (u: string, p: string) => Promise<void>; logout: () => void }
const Ctx = createContext<AuthState>({ user: null, loading: true, login: async () => {}, logout: () => {} })

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const onUnauthorized = () => setUser(null)
    window.addEventListener('eegdep:unauthorized', onUnauthorized)
    if (tokenStore.get()) api.me().then(setUser).catch(() => tokenStore.clear()).finally(() => setLoading(false))
    else setLoading(false)
    return () => window.removeEventListener('eegdep:unauthorized', onUnauthorized)
  }, [])

  const login = useCallback(async (username: string, password: string) => {
    const t = await api.login(username, password)
    tokenStore.set(t.access_token)
    setUser(t.user)
  }, [])
  const logout = useCallback(() => { tokenStore.clear(); setUser(null) }, [])

  return <Ctx.Provider value={{ user, loading, login, logout }}>{children}</Ctx.Provider>
}

export const useAuth = () => useContext(Ctx)
