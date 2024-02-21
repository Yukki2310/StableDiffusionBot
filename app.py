from os import getenv
import discord
from discord.ext import commands
import asyncio
import deepl

bot = commands.Bot(command_prefix='/', intents=discord.Intents.all())

##
## reaction
##
@bot.event
async def on_message(message: discord.Message):
    if (bot.user.mentioned_in(message)
            and message.mention_everyone is False 
            and message.type != discord.MessageType.reply):
        # show input indicator
        async with message.channel.typing():

            # generate Translator object
            trans = deepl.Translator(getenv('DEEPL_TOKEN'))
            # translate into English
            result = trans.translate_text(message.content, target_lang='EN-US')

            # wait image generation
            await generateImage("masterpiece, best quality, detailed, ultra hires, 8k, "+str(result))
        await message.reply(content="画像を生成しました。", file=discord.File("tmp/output.png"))

##
## generate Image from prompt
##
async def generateImage(prompt: str):
    command = 'modal run sdxl.py --prompt "{}"'.format(prompt)
    process = await asyncio.create_subprocess_shell(command)
    await process.communicate()

bot.run(getenv('SDXL_TOKEN'))