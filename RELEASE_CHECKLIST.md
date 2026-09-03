# Checklist публикации vmlib

Публичный репозиторий: <https://github.com/vm-course-2026/vmlib>.
Публикуется только результат строгого экспортера, а не родительский рабочий
каталог и не ранее собранный `dist/`.

## Проверка исходников

- [ ] Версии в `pyproject.toml` и `vmlib.__version__` совпадают.
- [ ] `uv lock --check` подтверждает актуальность `uv.lock`.
- [ ] `uv sync --frozen --extra test` завершается без изменения файлов.
- [ ] `MPLBACKEND=Agg uv run pytest` проходит полностью.
- [ ] `uv run python tools/build_release.py` выполняет две независимые сборки,
  требует совпадения SHA-256 и создаёт ровно один wheel и один sdist.
- [ ] `uv run python tools/build_release.py --check-only` подтверждает, что у
  всех tar-записей sdist владелец `0:0 root:root`, а время и порядок фиксированы.
- [ ] Wheel установлен в чистое окружение; импорт и `vmlib-doctor` работают.
- [ ] В исходниках нет токенов, персональных данных и закрытого кода.
- [ ] Закрытый release pipeline запускает `tools/build_vmlib_public.py` из
  корня сборочного проекта и получает чистый репозиторий с одним коммитом и
  пустым `git status`.

## Настройки GitHub

- [ ] Default branch — `main`; слияние только через pull request.
- [ ] Обязательный check branch ruleset — `quality-gate`.
- [ ] Для Actions установлены read-only permissions по умолчанию.
- [ ] Issues и Discussions включены.
- [ ] Private Vulnerability Reporting включён, а ссылка из `SECURITY.md`
  открывает приватную форму.
- [ ] Dependabot читает `uv.lock` и версии GitHub Actions.
- [ ] Созданы метки из шаблонов и Dependabot:
  `bug`, `needs-triage`, `needs-reproduction`, `documentation`,
  `good first issue`, `question`, `wontfix`, `dependencies`, `python`,
  `github-actions`.
- [ ] Issue form автоматически ставит существующие метки `bug` и
  `needs-triage`; приватные отчёты направляются в Security Advisories.

## Выпуск

- [ ] Изменения версии и release notes просмотрены вторым участником.
- [ ] Тег имеет вид `vX.Y.Z` и указывает на проверенный commit `main`.
- [ ] Опубликованный тег не существовал ранее: теги и артефакты старых релизов
  никогда не перемещаются и не заменяются.
- [ ] GitHub Release содержит sdist, wheel и контрольные SHA-256.
- [ ] Установка по новому тегу проверена в чистом окружении.
- [ ] Для patch-релиза перечислены затронутые ДЗ; если исправление влияет на
  уже выданную работу, её эталон и тесты проходят на обеих версиях либо
  различие явно описано.
- [ ] Update-PR для выданной ДЗ атомарно обновляет `assignment_version`,
  release manifest, `pyproject.toml` и `uv.lock`; старый grading profile
  сохранён для ранее зафиксированных попыток.
