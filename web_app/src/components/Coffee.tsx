import { Coffee as CoffeeIcon } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Dialog, DialogContent, DialogTitle, DialogTrigger } from "./ui/dialog"
import { IconButton } from "./ui/button"
import { DialogDescription } from "@radix-ui/react-dialog"
import Kofi from "@/assets/kofi_button_black.png"

export function Coffee() {
  const { t } = useTranslation()

  return (
    <Dialog>
      <DialogTrigger asChild>
        <IconButton tooltip={t("coffee.tooltip")}>
          <CoffeeIcon />
        </IconButton>
      </DialogTrigger>
      <DialogContent>
        <DialogTitle>{t("coffee.title")}</DialogTitle>
        <DialogDescription className="mb-8">
          {t("coffee.description")}
        </DialogDescription>
        <div className="w-full flex items-center justify-center">
          <a
            href="https://ko-fi.com/Z8Z1CZJGY"
            target="_blank"
            rel="noreferrer"
          >
            <img src={Kofi} className="h-[32px]" />
          </a>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export default Coffee
