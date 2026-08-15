import { useStore } from "@/lib/states"
import {
  useHotkeys,
  type HotkeyCallback,
  type Keys,
} from "react-hotkeys-hook"
import type { DependencyList } from "react"

const useHotKey = (
  keys: Keys,
  callback: HotkeyCallback,
  deps?: DependencyList
) => {
  const disableShortCuts = useStore((state) => state.disableShortCuts)
  return useHotkeys(keys, callback, { enabled: !disableShortCuts }, deps)
}

export default useHotKey
