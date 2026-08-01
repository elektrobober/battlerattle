// Хроника сессии — June-стиль: тёмная обложка, кремовые страницы,
// бордо/золото. Данные приходят из data.json (см. build_pdf_data).
#let data = json("data.json")

#let dark = rgb("#241812")
#let cream = rgb("#f4edd8")
#let gold = rgb("#b9974e")
#let bordo = rgb("#7a1f1f")
#let ink = rgb("#3a2a1a")
#let muted = rgb("#7a5c3a")

#set text(font: ("PT Serif", "Libertinus Serif"), size: 10.5pt, fill: ink, lang: "ru")

// ── Обложка ──
#set page(paper: "a4", fill: dark, margin: (x: 2.2cm, top: 3.5cm, bottom: 2.5cm))
#align(center)[
  #text(fill: gold, style: "italic", size: 13pt)[#data.subtitle]
  #v(0.6em)
  #text(fill: cream, size: 30pt, weight: "bold")[ХРОНИКА СЕССИИ]
  #v(0.2em)
  #text(fill: gold, size: 26pt, weight: "bold")[#data.session]
  #v(1.2em)
  #if data.scenes.len() > 0 and data.scenes.at(0).file != none [
    #image(data.scenes.at(0).file, width: 92%, height: 45%, fit: "contain")
  ]
  #v(1em)
  #if data.scenes.len() > 0 [
    #text(fill: bordo.lighten(35%), style: "italic", size: 12pt)[#data.scenes.at(0).title]
  ]
]

// ── Внутренние страницы ──
#set page(
  fill: cream,
  margin: (x: 2.2cm, top: 2.6cm, bottom: 2.4cm),
  background: pad(0.9cm, rect(width: 100%, height: 100%,
    stroke: 1pt + gold.darken(20%), radius: 1pt,
    inset: 3pt, rect(width: 100%, height: 100%, stroke: 0.5pt + gold))),
  footer: context align(center,
    text(size: 8pt, style: "italic", fill: muted)[
      #data.campaign_title · Сессия #data.session · стр. #counter(page).display()
    ]),
)
#counter(page).update(1)
#set heading(numbering: none)
#show heading.where(level: 1): it => [
  #text(fill: bordo, size: 18pt, weight: "bold")[#it.body]
  #v(-0.4em)
  #line(length: 100%, stroke: 0.7pt + gold.darken(10%))
  #v(0.4em)
]
#show heading.where(level: 2): it => text(fill: bordo, size: 13pt, weight: "bold")[#it.body]

#pagebreak(weak: true)
= Оглавление
#outline(title: none, depth: 1, indent: auto)

#pagebreak()
= Сводка
#text(style: "italic")[Сессия в цифрах: #data.summaries.len() эпизодов · #data.mvp_events.len() MVP-сигналов · #data.dice.len() бросков.]
#v(0.6em)
== MVP — кто затащил сессию
#{
  let max_score = if data.mvp_scores.len() > 0 { calc.max(..data.mvp_scores.map(s => s.score)) } else { 1 }
  table(
    columns: (auto, 1fr, auto), stroke: none, row-gutter: 0.45em,
    ..data.mvp_scores.enumerate().map(((i, s)) => (
      [#text(fill: if i == 0 { gold.darken(20%) } else { bordo })[#(i + 1). #s.character]],
      [#rect(width: 100% * s.score / max_score, height: 7pt,
             fill: if i == 0 { gold.darken(10%) } else { bordo }, radius: 3pt)],
      [*#s.score*],
    )).flatten()
  )
}
#v(0.6em)
== Кости — удача за столом
#table(
  columns: (1fr, auto, auto, auto, auto),
  stroke: 0.4pt + gold.darken(10%), inset: 6pt,
  table.header([*Персонаж*], [*Средний d20*], [*Бросков*], [*нат-20*], [*нат-1*]),
  ..data.dice_stats.map(s => (
    [#s.character],
    [#calc.round(s.avg, digits: 2)],
    [#s.count],
    [#if s.nat20 > 0 [#text(fill: gold.darken(20%))[*#s.nat20*]] else [·]],
    [#if s.nat1 > 0 [#text(fill: bordo)[*#s.nat1*]] else [·]],
  )).flatten()
)

#if data.recap != "" [
  #pagebreak()
  = Рекап — что было в прошлый раз
  #text(style: "italic")[Краткий пересказ ключевых событий сессии.]
  #v(0.5em)
  #for para in data.recap.split("\n\n") [
    #par(justify: true)[#para]
  ]
]

#if data.quest_hooks.len() > 0 [
  #pagebreak()
  = Зацепки и квесты
  #text(style: "italic")[Самое важное для следующей игры.]
  #v(0.5em)
  #for hook in data.quest_hooks [
    #par[— *#hook.title.* #hook.description]
    #v(0.3em)
  ]
]

#if data.party.len() > 0 [
  #pagebreak()
  = Партия
  #grid(
    columns: (1fr, 1fr), gutter: 1.2em,
    ..data.party.map(m => align(center)[
      #if m.ref_file != none [#image(m.ref_file, width: 100%)]
      #text(fill: bordo, weight: "bold")[#m.name]
      #if "class_ru" in m [ \ #text(style: "italic", size: 9pt)[#m.class_ru] ]
      #if "player" in m [ \ #text(style: "italic", size: 9pt)[игрок: #m.player] ]
    ])
  )
]

#if data.scenes.len() > 1 [
  #pagebreak()
  = Ключевые сцены
  #for scene in data.scenes.slice(1) [
    #block(breakable: false)[
      #if scene.file != none [#image(scene.file, width: 100%, height: 11cm, fit: "contain")]
      #align(center)[#text(fill: bordo, weight: "bold")[#scene.title] #text(size: 9pt, fill: muted)[#raw(scene.time)]]
    ]
    #v(0.8em)
  ]
]

#pagebreak()
= Хроника сессии — полный ход
#for s in data.summaries [
  #par[*Эпизод #s.chunk_index.* #s.summary]
  #v(0.35em)
]

#pagebreak()
= MVP — полный разбор
#for e in data.mvp_events [
  #par(hanging-indent: 1em)[#raw(e.time) *#e.character* +#e.weight [#e.category] — #e.reason]
]

#pagebreak()
= Кости — полная выкладка
#for d in data.dice [
  #par(hanging-indent: 1em)[#raw(d.time) *#d.character* #d.roll_type: нат=#repr(d.natural) мод=#repr(d.modifier) итог=#repr(d.total) — #d.context]
]

#pagebreak()
= Тайм-лайн ключевых действий
#for a in data.actions [
  #par(hanging-indent: 1em)[#raw(a.time) *#a.character* — #a.action → #a.outcome]
]
