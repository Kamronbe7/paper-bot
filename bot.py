# bot.py
from __future__ import annotations
import os, asyncio, tempfile
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
)
from aiogram.filters import CommandStart, Command

from storage import MemoryState
from rag import PaperRAG

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN .env da yo‘q")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

state = MemoryState()
rag = PaperRAG()

# -------------------- START --------------------
@dp.message(CommandStart())
async def start(m: Message):
    await m.answer(
        "👋 Salom!\n\n"
        "📄 PDF yuboring — men uni tahlil qilaman.\n\n"
        "Buyruqlar:\n"
        "• /summary — xulosa\n"
        "• /ask <savol> — savol berish\n"
        "• /status — holat\n"
    )

# -------------------- PDF HANDLER --------------------
@dp.message(F.document)
async def handle_pdf(m: Message):
    doc = m.document
    if not doc.file_name.lower().endswith(".pdf"):
        await m.answer("❌ Faqat PDF qabul qilinadi.")
        return

    await m.answer("⏳ PDF qabul qilindi. Processing boshlanmoqda...")

    file = await bot.get_file(doc.file_id)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    await bot.download_file(file.file_path, tmp_path)

    user_id = m.from_user.id
    state.set_status(user_id, "processing")

    async def background_job():
        try:
            paper_id = rag.ingest_pdf(tmp_path, doc.file_name)
            state.set_active(user_id, paper_id, doc.file_name)
            state.set_status(user_id, "ready")

            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📄 Summary", callback_data="summary")],
                [InlineKeyboardButton(text="❓ Sample questions", callback_data="samples")]
            ])
            await bot.send_message(
                user_id,
                f"✅ Tayyor! Aktiv hujjat: {doc.file_name}",
                reply_markup=kb
            )
        except Exception as e:
            state.set_status(user_id, "error")
            await bot.send_message(user_id, f"❌ Xatolik: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    asyncio.create_task(background_job())

# -------------------- STATUS --------------------
@dp.message(Command("status"))
async def status(m: Message):
    st = state.get_status(m.from_user.id)
    if not st:
        await m.answer("ℹ️ Hozircha hech narsa yuklanmagan.")
    elif st == "processing":
        await m.answer("⏳ PDF tahlil qilinmoqda...")
    elif st == "ready":
        await m.answer("✅ PDF tayyor.")
    else:
        await m.answer("❌ Xatolik bo‘lgan.")

# -------------------- SUMMARY --------------------
@dp.message(Command("summary"))
async def summary(m: Message):
    st = state.get_active(m.from_user.id)
    if not st:
        await m.answer("📄 Avval PDF yuboring.")
        return

    await m.answer("🧠 Xulosa tayyorlanmoqda...")
    text = await asyncio.to_thread(rag.summarize, st.paper_id)
    for part in split(text):
        await m.answer(part)

# -------------------- ASK --------------------
@dp.message(Command("ask"))
async def ask(m: Message):
    lang = state.get_lang(user_id)
    res = await asyncio.to_thread(rag.ask, paper.paper_id, q, lang)
    await m.answer(res["answer"])

    if res["sources"]:
        s = "📌 Sources:\n" + "\n".join(
            [f"[{x['ref']}] chunk {x['chunk_no']}/{x['total_chunks']} — {x['snippet']}" for x in res["sources"]]
        )
        await m.answer(s)

# -------------------- CALLBACKS --------------------
@dp.callback_query(F.data == "summary")
async def cb_summary(c: CallbackQuery):
    await summary(c.message)
    await c.answer()

@dp.callback_query(F.data == "samples")
async def cb_samples(c: CallbackQuery):
    await c.message.answer(
        "❓ Namuna savollar:\n"
        "• /ask Maqolaning asosiy maqsadi nima?\n"
        "• /ask Qanday metod ishlatilgan?\n"
        "• /ask Natijalar qanday?\n"
        "• /ask Future work nimalar?"
    )
    await c.answer()

# -------------------- UTILS --------------------
def split(text: str, n: int = 3500):
    out, cur = [], ""
    for line in text.splitlines(True):
        if len(cur) + len(line) > n:
            out.append(cur)
            cur = ""
        cur += line
    if cur:
        out.append(cur)
    return out

# -------------------- RUN --------------------
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
