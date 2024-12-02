from rich.console import Console
from rich.markdown import Markdown
from langchain_core.messages import SystemMessage

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from ethics.settings import settings

console = Console()

MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.7

CASE_DESCRIPTION = """
ИИ-система, которая будет способна анализировать слайды презентаций, находить в них проблемы и предлагать улучшения, или просто модифицировать один или несколько слайдов по запросу пользователя.
Нужно ответить на вопрос: кто несет ответственность за изменения в слайдах, сделанные ИИ, которые могли ввести публику в заблуждение?
"""

PARAPGRAPH_DESCRIPTION_BY_ID = {
    1: """
    Формулировка этической проблемы, которая связана с разбираемой ситуацией, или ответ на вопрос, в чем заключается этическая проблема в анализируемой ситуации?
    Что представляется неоднозначным с позиции этического анализа?
    """,
    2: """
    Выявление всех заинтересованных лиц, которых касается рассматриваемая ситуация (это могут люди с определенными функциями (пользователь, разработчик, маркетолог), социальные группы по уровню доходу, возрасту, гендеру и т.п., другие живые существа, все человечество, все живые существа на планете и т.п.).
    Распределение ролей: кто - моральный агент=субъект (агенты, субъекты) и кто – объект (объекты) морального воздействия.
    Определение мотивов/намерений, заинтересованных лиц.
    Определение моральных ценностей, затрагиваемых в ситуации (например: автономия человека, отсутствие неявной дискриминации, жизнь и здоровье человека и т.д.)
    """,
    3: """
    Определение нескольких возможных сценариев оценки ситуации (возможных вариантов ответа на вопрос “как правильно”) с соответствующей каждому обоснованию, аргументации. С учетом того, что мы различаем этический и правовой аспекты и исследуем этический аспект.
    """,
    4: """
    Определение стоящих за сценариями комплексов явно или неявно признаваемых ценностей 
    (например: безопасность общества vs неприкосновенность частной жизни, развитие технологий vs социальная справедливость).
    """,
    5: """
    Выбор наилучшего, с точки зрения автора, сценария оценки с обоснованием данного выбора.
    """,
}


@tool
def read_theory_doc() -> str:
    """Прочитать документ с теорией по этике в сфере ИИ"""

    with open("ethics/data/theory.md", "r", encoding="utf-8") as f:
        return f.read()


@tool
def read_task_doc() -> str:
    """Прочитать документ с требованиями ко всей структуре эссе"""

    with open("ethics/data/task.md", "r", encoding="utf-8") as f:
        return f.read()


def create_section_writer(llm, section_number):
    agent = create_react_agent(
        llm,
        tools=[read_theory_doc],
        state_modifier=f"""Вы - профессиональный агент по написанию эссе по этике в сфере ИИ, сейчас вы пишете {section_number}-й раздел.

        ## Описание ситуации для эссе

        {CASE_DESCRIPTION}
        
        ## Требования к содержанию раздела номер {section_number}

        {PARAPGRAPH_DESCRIPTION_BY_ID[section_number]}

        ## Инструкции

        1. Пишите раздел с учетом предыдущего контекста.
        2. Текст должен быть академическим, хорошо структурированным и аргументированным. Объясняйте свои рассуждения.
        3. Содержимое раздела должно относиться именно к технологии ИИ, которая анализирует презентации и предлагает улучшения.
        4. Не используйте общие фразы вида "в данном разделе мы рассмотрим...", "в этом разделе мы обсудим..." и т.п, пишите именно по текущему разделу.
        5. Старайтесь писать короткими предложениями, не используя сложные конструкции. Помните, что длина всего эссе должна быть 2000-3000 знаков.""",
    )

    return agent


def invoke_writer(writer, current_essay: str = None) -> str:
    """Общая функция для вызова writer'а и форматирования результата"""
    messages = []
    if current_essay:
        messages.append(SystemMessage(content=f"Текущая версия эссе: {current_essay}"))

    result = writer.invoke({"messages": messages})
    return result["messages"][-1].content + "\n\n"


def create_essay_refiner(llm):
    return create_react_agent(
        llm,
        tools=[read_theory_doc, read_task_doc],
        state_modifier=f"""Вы - профессиональный агент по написанию эссе по этике в сфере ИИ.
        Вам показана предварительная версия эссе, которую вы должны доработать.
        
        ## Описание ситуации для эссе

        {CASE_DESCRIPTION}
        
        ## Инструкции
        1. Внимательно ознакомьтесь с теорией и требованиями к структуре эссе.
        2. Не меняйте структуру, сохраните порядок разделов.
        3. Объедините все части в единый текст, внесите корректировки, если они необходимы.
        4. Проверьте целостность текста, логичность переходов между частями, соответствие требованиям.
        5. Напишите финальную версию эссе. Она должна быть 2000-3000 знаков.""",
    )


def write_essay():
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )

    section1 = invoke_writer(create_section_writer(llm, 1))
    console.print(Markdown(f"### Раздел 1\n{section1}"))

    current_essay = section1

    section2 = invoke_writer(create_section_writer(llm, 2), current_essay)
    console.print(Markdown(f"### Раздел 2\n{section2}"))

    current_essay += section2

    section3 = invoke_writer(create_section_writer(llm, 3), current_essay)
    console.print(Markdown(f"### Раздел 3\n{section3}"))

    current_essay += section3

    section4 = invoke_writer(create_section_writer(llm, 4), current_essay)
    console.print(Markdown(f"### Раздел 4\n{section4}"))

    current_essay += section4

    section5 = invoke_writer(create_section_writer(llm, 5), current_essay)
    console.print(Markdown(f"### Раздел 5\n{section5}"))

    current_essay += section5

    console.print(Markdown(f"### Предварительная версия эссе\n{current_essay}"))

    essay_refiner = create_essay_refiner(llm)

    refined_essay = invoke_writer(essay_refiner, current_essay)
    console.print(Markdown(f"### Финальная версия эссе\n{refined_essay}"))

    return refined_essay


def main():
    essay = write_essay()

    with open("ethics/data/essay.txt", "w", encoding="utf-8") as f:
        f.write(essay)
    print()
    print("Эссе успешно сохранено в файл 'ethics/data/essay.txt'")


if __name__ == "__main__":
    main()
