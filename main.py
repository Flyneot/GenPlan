import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from collections import Counter
from datetime import datetime


# Генерация эпиков с случайными данными
epics = []
num_epics = 10

for i in range(1, num_epics + 1):
    epic = {
        "name": f"Epic{i}",
        "backend_hours": random.randint(16, 80),
        "frontend_hours": random.randint(8, 60),
        "design_hours": random.randint(4, 20),
        "analytics_hours": random.randint(4, 20),
        "testing_hours": random.randint(8, 60),
        "utility": random.randint(1, 10),
        "dependencies": []
    }
    # Добавляем зависимости с вероятностью 30%
    if i > 1 and random.random() < 0.3:
        dep_index = random.randint(1, i - 1)
        epic["dependencies"].append(f"Epic{dep_index}")
    epics.append(epic)

# Список сотрудников с их ролями
employees = [
    {"id": 1, "role": "backend"},
    {"id": 2, "role": "backend"},
    {"id": 3, "role": "backend"},
    {"id": 4, "role": "frontend"},
    {"id": 5, "role": "frontend"},
    {"id": 6, "role": "design"},
    {"id": 7, "role": "analytics"},
    {"id": 8, "role": "testing"},
    {"id": 9, "role": "testing"}
]

# Вычисляем количество сотрудников по ролям
role_counts = Counter(emp['role'] for emp in employees)

release_days = 60  # Увеличиваем длительность релиза
release_hours = release_days * 8  # Максимальное количество часов для релиза

# Отклонения по ролям (для учета неопределенности)
deviations = {
    "backend": 0.1,
    "frontend": 0.1,
    "design": 0.1,
    "analytics": 0.1,
    "testing": 0.1
}

# Фиксируем случайные семена для воспроизводимости
random.seed(42)
np.random.seed(42)

# Создаем типы для NSGA-II
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Максимизируем utility, минимизируем риск
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(epics))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def calculate_hours_with_uncertainty(hours, deviation): # Робастая оптимизация
    adjusted_hours = np.random.normal(hours, hours * deviation)
    adjusted_hours = max(0, adjusted_hours)  # Ограничиваем снизу нулём
    return adjusted_hours

def evaluate_individual(individual):
    total_utility = 0
    role_hours = {role: 0 for role in deviations.keys()}
    penalty = 0
    total_risk = 0

    for i, selected in enumerate(individual):
        if selected:
            epic = epics[i]
            epic_possible = True
            epic_role_hours = {}

            # Используем детерминированные оценки часов для предварительной оценки
            for role in deviations.keys():
                role_key = f"{role}_hours"
                hours = epic[role_key]
                epic_role_hours[role] = hours

                # Проверяем, не превышает ли выполнение эпика доступные ресурсы по роли
                if role_hours[role] + hours > role_counts[role] * release_hours:
                    epic_possible = False
                    break  # Нет смысла проверять дальше

            if epic_possible:
                # Эпик может быть выполнен, обновляем суммарные часы
                for role in deviations.keys():
                    role_hours[role] += epic_role_hours[role]
                    # Рассчитываем риск как сумма отклонений по ролям
                    total_risk += epic_role_hours[role] * deviations[role]

                total_utility += epic["utility"]

                # Штраф за невыполненные зависимости
                for dependency in epic.get('dependencies', []):
                    dep_index = next((j for j, e in enumerate(epics) if e["name"] == dependency), None)
                    if dep_index is not None and individual[dep_index] == 0:
                        penalty += 1000  # Штраф за невыполненную зависимость
            else:
                # Эпик не может быть выполнен, исключаем его
                individual[i] = 0  # Отменяем выбор эпика
                penalty += 1000  # Штраф за попытку выбрать невыполнимый эпик

    # Функция приспособленности для многокритериальной оптимизации
    # Цели: максимизировать полезность (с учетом штрафов), минимизировать риск
    fitness = (total_utility - penalty, total_risk)
    return fitness

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selNSGA2)  # Используем NSGA-II

def run_epic_selection_nsga2(pop_size, ngen, cxpb, mutpb):
    pop = toolbox.population(n=pop_size)
    # Инициализируем популяцию и вычисляем приспособленность
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Инициализируем атрибуты crowding_dist
    pop = toolbox.select(pop, len(pop))

    # Алгоритм NSGA-II
    for gen in range(ngen):
        # Селекция родителей
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Применение операторов кроссовера и мутации
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() <= mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Оценка новых особей
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Объединяем популяции и применяем селекцию NSGA-II
        pop = toolbox.select(pop + offspring, pop_size)

    # Получаем неплохие решения из последней популяции
    pareto_front = tools.sortNondominated(pop, k=pop_size, first_front_only=True)[0]
    return pareto_front

# Создаем типы для генетического алгоритма расписания
creator.create("ScheduleFitness", base.Fitness, weights=(-1.0,))  # Минимизируем общее время выполнения и простой
creator.create("ScheduleIndividual", dict, fitness=creator.ScheduleFitness)

schedule_toolbox = base.Toolbox()

def plan_task(epic, role, earliest_start, schedule):
    role_key = f"{role}_hours"
    hours = epic.get(role_key, 0)
    duration = int(np.ceil(hours / 8))

    if duration <= 0 or duration > release_days:
        print(f"Задача {role} для эпика {epic['name']} имеет некорректную длительность: {duration} дней")
        return None

    role_employees = [emp for emp in employees if emp['role'] == role]
    if not role_employees:
        print(f"Нет доступных сотрудников с ролью {role} для эпика {epic['name']}")
        return None

    # Попытка найти день начала задачи начиная с earliest_start и далее
    # для минимизации разрыва между этапами
    best_employee = None
    best_start_day = None
    best_gap = float('inf')

    # Будем искать подходящий день начиная с earliest_start до конца доступного релиза
    for candidate_start in range(earliest_start, release_days - duration + 1):
        # Для каждого сотрудника проверим, может ли он взять эту задачу с candidate_start
        for emp in role_employees:
            emp_id = emp['id']
            # Проверяем занятость сотрудника
            busy = False
            for e_name, tasks in schedule.items():
                for t in tasks:
                    if t['employee_id'] == emp_id:
                        t_start = t['start_day']
                        t_end = t_start + t['duration']
                        # Проверяем пересечение интервалов [candidate_start, candidate_start+duration)
                        if not (candidate_start + duration <= t_start or candidate_start >= t_end):
                            busy = True
                            break
                if busy:
                    break

            if not busy:
                # Сотрудник свободен, считаем "разрыв" относительно earliest_start
                gap = candidate_start - earliest_start
                if gap < best_gap:
                    best_gap = gap
                    best_start_day = candidate_start
                    best_employee = emp

        # Если мы уже нашли оптимальный вариант на этот день (gap = 0 это идеально),
        # то нет смысла искать дальше
        if best_gap == 0:
            break

    if best_employee is None:
        # Не смогли найти сотрудника под эту задачу
        return None

    task = {
        'epic': epic['name'],
        'employee_id': best_employee['id'],
        'start_day': best_start_day,
        'duration': duration,
        'role': role
    }
    return task, best_start_day + duration


def generate_schedule_individual():
    selected_epics = schedule_toolbox.selected_epics
    schedule = {}

    for epic in selected_epics:
        epic_tasks = []
        success = True
        previous_end = 0

        stages = [
            {'roles': ['design'], 'type': 'sequential'},
            {'roles': ['analytics'], 'type': 'sequential'},
            {'roles': ['backend', 'frontend'], 'type': 'parallel'},
            {'roles': ['testing'], 'type': 'sequential'}
        ]

        for stage in stages:
            stage_start = previous_end
            stage_end = stage_start

            if stage['type'] == 'sequential':
                role = stage['roles'][0]
                task_info = plan_task(epic, role, stage_start, schedule)
                if not task_info:
                    success = False
                    break
                task, task_end = task_info
                epic_tasks.append(task)
                stage_end = task_end
            elif stage['type'] == 'parallel':
                max_task_end = stage_start
                parallel_tasks = []
                for role in stage['roles']:
                    task_info = plan_task(epic, role, stage_start, schedule)
                    if not task_info:
                        success = False
                        break
                    task, task_end = task_info
                    parallel_tasks.append(task)
                    if task_end > max_task_end:
                        max_task_end = task_end
                if not success:
                    break
                epic_tasks.extend(parallel_tasks)
                stage_end = max_task_end
            else:
                success = False
                print(f"Неизвестный тип этапа: {stage['type']}")
                break

            previous_end = stage_end

        # Если эпик спланирован успешно, добавляем в расписание
        if success and epic_tasks:
            schedule[epic['name']] = epic_tasks
        else:
            # Если не получилось спланировать эпик, просто игнорируем его
            # и переходим к следующему эпику, не прерывая цикл
            print(f"Эпик {epic['name']} не может быть полностью запланирован. Пропускаем.")

    return creator.ScheduleIndividual(schedule)


def evaluate_schedule(individual):
    total_duration = 0
    penalty = 0
    total_idle_time = 0

    if not individual:
        return (float('inf'),)

    employee_calendar = {emp['id']: [0] * release_days for emp in employees}
    epic_dict = {epic['name']: epic for epic in epics}

    # Коэффициенты штрафов (оставим ранее сниженные)
    dependency_penalty = 100
    partial_epic_penalty = 100
    sequence_penalty = 50
    idle_break_penalty = 10
    resource_exceed_penalty = 200

    # Считаем суммарную полезность
    total_utility = 0

    for epic_name, tasks in individual.items():
        epic = epic_dict.get(epic_name)
        if not epic:
            continue

        # Добавляем полезность эпика, только если он полным набором ролей
        required_roles = {'design', 'analytics', 'backend', 'frontend', 'testing'}
        task_roles = {t['role'] for t in tasks}
        if not required_roles.issubset(task_roles):
            penalty += partial_epic_penalty
            continue
        else:
            # Эпик полностью запланирован - добавляем его полезность
            total_utility += epic["utility"]

        # Проверка зависимостей
        for dependency in epic.get('dependencies', []):
            if dependency not in individual:
                penalty += dependency_penalty
            else:
                dep_tasks = individual[dependency]
                dep_end = max(t['start_day'] + t['duration'] for t in dep_tasks)
                epic_start = min(t['start_day'] for t in tasks)
                if dep_end > epic_start:
                    penalty += dependency_penalty

        # Проверяем последовательность задач
        tasks_by_role = {t['role']: t for t in tasks}
        design_task = tasks_by_role['design']
        analytics_task = tasks_by_role['analytics']
        backend_task = tasks_by_role['backend']
        frontend_task = tasks_by_role['frontend']
        testing_task = tasks_by_role['testing']

        design_end = design_task['start_day'] + design_task['duration']
        analytics_start = analytics_task['start_day']
        if analytics_start < design_end:
            penalty += sequence_penalty
        else:
            design_to_analytics_break = analytics_start - design_end
            penalty += design_to_analytics_break * idle_break_penalty

        analytics_end = analytics_task['start_day'] + analytics_task['duration']
        backend_start = backend_task['start_day']
        frontend_start = frontend_task['start_day']
        if backend_start < analytics_end or frontend_start < analytics_end:
            penalty += sequence_penalty
        else:
            analytics_to_dev_break = (backend_start - analytics_end) + (frontend_start - analytics_end)
            penalty += analytics_to_dev_break * idle_break_penalty

        backend_end = backend_task['start_day'] + backend_task['duration']
        frontend_end = frontend_task['start_day'] + frontend_task['duration']
        dev_end = max(backend_end, frontend_end)
        testing_start = testing_task['start_day']
        if testing_start < dev_end:
            penalty += sequence_penalty
        else:
            dev_to_testing_break = testing_start - dev_end
            penalty += dev_to_testing_break * idle_break_penalty

        # Проверка занятости сотрудников и выхода за пределы релиза
        for task in tasks:
            employee_id = task['employee_id']
            start_day = task['start_day']
            duration = task['duration']
            role = task['role']

            deviation = deviations[role]
            adjusted_duration = int(np.ceil(duration * (1 + deviation)))
            end_day = start_day + adjusted_duration

            if end_day > release_days:
                penalty += resource_exceed_penalty
                continue

            if any(employee_calendar[employee_id][d] == 1 for d in range(start_day, end_day)):
                penalty += resource_exceed_penalty
                continue
            else:
                for d in range(start_day, end_day):
                    employee_calendar[employee_id][d] = 1
                total_duration = max(total_duration, end_day)

    # Подсчитываем простой сотрудников
    for emp_id, calendar in employee_calendar.items():
        total_idle_time += calendar.count(0)

    # Включаем полезность в фитнес
    # Чем больше total_utility, тем сильнее уменьшается фитнес,
    # стимулируя алгоритм к планированию эпиков с максимальной суммарной полезностью.
    utility_factor = 100  # Можно подбирать
    fitness = (total_duration + penalty + total_idle_time - utility_factor * total_utility,)

    return fitness


def evaluate_solution(epics_list, schedule_individual):
    total_utility = sum(e['utility'] for e in epics_list)
    if schedule_individual.fitness.valid:
        schedule_fitness = schedule_individual.fitness.values[0]
    else:
        schedule_fitness = float('inf')

    score = total_utility * 10- schedule_fitness * 0.5
    return score


# Определяем пользовательские функции кроссовера и мутации
def cxTwoPointDict(ind1, ind2):
    keys1 = list(ind1.keys())
    keys2 = list(ind2.keys())
    all_keys = list(set(keys1 + keys2))
    if len(all_keys) < 2:
        return ind1, ind2  # Недостаточно ключей для кроссовера

    # Выбираем две точки кроссовера
    cxpoint1 = random.randint(0, len(all_keys) - 2)
    cxpoint2 = random.randint(cxpoint1 + 1, len(all_keys) - 1)

    # Обмен значениями между точками кроссовера
    for key in all_keys[cxpoint1:cxpoint2]:
        val1 = ind1.get(key, None)
        val2 = ind2.get(key, None)

        if val1 is not None:
            ind2[key] = val1
        else:
            ind2.pop(key, None)

        if val2 is not None:
            ind1[key] = val2
        else:
            ind1.pop(key, None)

    return ind1, ind2


def mutShuffleDict(individual, indpb):
    """Выполняет мутацию индивидуумов-словарей."""
    for epic_name in list(individual.keys()):
        if random.random() < indpb:
            tasks = individual.get(epic_name)
            if tasks is None:
                # С вероятностью 0.5 пытаемся добавить эпик обратно
                if random.random() < 0.5:
                    new_tasks = generate_tasks_for_epic(epic_name)
                    if new_tasks:
                        individual[epic_name] = new_tasks
            else:
                # Мутируем существующие задачи
                for task in tasks:
                    duration = task['duration']
                    task['start_day'] = random.randint(0, release_days - duration)
                    if random.random() < 0.5:
                        role_employees = [emp for emp in employees if emp['role'] == task['role']]
                        if role_employees:
                            task['employee_id'] = random.choice(role_employees)['id']
                # С вероятностью 0.1 удаляем эпик из расписания
                if random.random() < 0.1:
                    individual.pop(epic_name)
    return individual,

def generate_tasks_for_epic(epic_name):
    """Пытается сгенерировать задачи для заданного эпика."""
    epic = next((e for e in schedule_toolbox.selected_epics if e['name'] == epic_name), None)
    if not epic:
        return None  # Эпик не найдена

    epic_tasks = []
    previous_end = 0  # Начальное время для дизайна
    roles_sequence = ['design', 'analytics', 'backend', 'frontend', 'testing']

    for role in roles_sequence:
        role_key = f"{role}_hours"
        adjusted_hours = calculate_hours_with_uncertainty(epic[role_key], deviations[role])
        duration = int(np.ceil(adjusted_hours / 8))

        start_day_min = previous_end
        start_day_max = release_days - duration
        if start_day_min > start_day_max:
            return None  # Невозможно запланировать задачу

        if start_day_min == start_day_max:
            start_day = start_day_min
        else:
            start_day = random.randint(start_day_min, start_day_max)

        role_employees = [emp for emp in employees if emp['role'] == role]
        if not role_employees:
            return None  # Нет сотрудников с этой ролью

        employee = random.choice(role_employees)
        task = {
            'epic': epic_name,
            'employee_id': employee['id'],
            'start_day': start_day,
            'duration': duration,
            'role': role
        }
        epic_tasks.append(task)
        previous_end = task['start_day'] + task['duration']

    return epic_tasks

schedule_toolbox.register("individual", generate_schedule_individual)
schedule_toolbox.register("population", tools.initRepeat, list, schedule_toolbox.individual)
schedule_toolbox.register("evaluate", evaluate_schedule)
schedule_toolbox.register("mate", cxTwoPointDict)
schedule_toolbox.register("mutate", mutShuffleDict, indpb=0.2)
schedule_toolbox.register("select", tools.selTournament, tournsize=3)

def run_schedule_ga(pop_size, ngen, cxpb, mutpb, selected_epics):
    # Сохраняем selected_epics внутри schedule_toolbox
    schedule_toolbox.selected_epics = selected_epics

    pop = schedule_toolbox.population(n=pop_size)

    # Инициализируем популяцию и вычисляем приспособленность
    fitnesses = list(map(schedule_toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Алгоритм генетического алгоритма
    for gen in range(ngen):
        offspring = schedule_toolbox.select(pop, len(pop))
        offspring = [schedule_toolbox.clone(ind) for ind in offspring]

        # Применение операторов кроссовера и мутации
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                schedule_toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() <= mutpb:
                schedule_toolbox.mutate(mutant)
                del mutant.fitness.values

        # Оценка новых особей
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(schedule_toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Обновление популяции
        pop[:] = offspring

    # Получаем лучшее решение
    best_individual = tools.selBest(pop, 1)[0]
    return best_individual

def visualize_schedule(schedule):
    import matplotlib.pyplot as plt

    selected_epics = schedule_toolbox.selected_epics

    # Подготовка данных для визуализации
    fig, ax = plt.subplots(figsize=(12, 8))

    yticks = []
    yticklabels = []
    y = 10  # Начальная позиция по оси Y

    # Создаем календарь занятости сотрудников
    employee_tasks = {emp['id']: [] for emp in employees}

    for epic_name, tasks in schedule.items():
        for task in tasks:
            employee_id = task['employee_id']
            start_day = task['start_day']
            duration = task['duration']
            end_day = start_day + duration
            employee_tasks[employee_id].append({
                'start_day': start_day,
                'duration': duration,
                'epic': epic_name,
                'role': task['role']
            })

    for emp in employees:
        tasks = employee_tasks[emp['id']]
        # Рисуем задачи сотрудника
        for task in tasks:
            ax.broken_barh([(task['start_day'], task['duration'])], (y, 9), facecolors=('tab:blue'))
            ax.text(task['start_day'] + task['duration']/2, y + 4.5, f"{task['epic']} ({task['role']})", ha='center', va='center', color='white')

        yticks.append(y + 4.5)
        yticklabels.append(f"Employee {emp['id']} ({emp['role']})")
        y += 10

    ax.set_ylim(0, y)
    ax.set_xlim(0, release_days)
    ax.set_xlabel('Дни')
    ax.set_ylabel('Сотрудники')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.grid(True)

    plt.title('Диаграмма Ганта расписания по сотрудникам')
    plt.tight_layout()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gantt_chart_{current_time}.png"
    # Сохраняем диаграмму в файл
    plt.savefig(filename)
    plt.show()

def iterative_optimization(max_iterations=5, improvement_threshold=0.01):
    # Начинаем с полного набора эпиков
    current_epics = epics[:]
    best_score = float('-inf')
    best_solution = None
    best_epics_set = None

    for iteration in range(max_iterations):
        print(f"Итерация {iteration+1}")

        # Запускаем NSGA-II для выбора эпиков
        pareto_front = run_epic_selection_nsga2(pop_size=300, ngen=200, cxpb=0.7, mutpb=0.2)
        best_individual = max(pareto_front, key=lambda ind: (ind.fitness.values[0], -ind.fitness.values[1]))
        selected_epics_indices = [i for i, gene in enumerate(best_individual) if gene == 1]
        selected_epics = [epics[i] for i in selected_epics_indices]

        if not selected_epics:
            print("Не выбрано ни одного эпика. Завершаем.")
            break

        # Составляем расписание для выбранных эпиков
        best_schedule = run_schedule_ga(pop_size=300, ngen=200, cxpb=0.7, mutpb=0.2, selected_epics=selected_epics)

        # Определяем, для каких эпиков удалось составить расписание
        scheduled_epic_names = set(best_schedule.keys())
        successfully_scheduled_epics = [e for e in selected_epics if e['name'] in scheduled_epic_names]

        # Оцениваем решение
        current_score = evaluate_solution(successfully_scheduled_epics, best_schedule)
        print(f"Оценка решения на итерации {iteration+1}: {current_score}")

        if best_score != float('-inf'):
            improvement = (current_score - best_score) / (abs(best_score) + 1e-9)
        else:
            improvement = 1.0

        if improvement < improvement_threshold:
            print("Улучшение незначительно, завершаем оптимизацию.")
            break

        # Обновляем лучшие решения
        best_score = current_score
        best_solution = best_schedule
        best_epics_set = successfully_scheduled_epics

        # Обновляем текущий набор эпиков - работаем только с теми, для кого удалось составить расписание
        current_epics = successfully_scheduled_epics
        # Для следующей итерации заменяем глобальный список epics на current_epics, если хотите
        # Но лучше просто использовать current_epics непосредственно при выборе, дописав Ваш NSGA-II для работы
        # с произвольным подмножеством эпиков.
        # Сейчас NSGA-II запускается на полный набор, можно изменить логику, чтобы NSGA-II тоже учитывал current_epics.
        # Для упрощения примера оставим как есть.

    return best_epics_set, best_solution, best_score

def main():
    final_epics, final_schedule, final_score = iterative_optimization(max_iterations=5, improvement_threshold=0.01)
    if final_epics and final_schedule:
        print("\nИтоговый набор эпиков:")
        for e in final_epics:
            print(f"- {e['name']} (utility={e['utility']})")

        print(f"\nИтоговый score: {final_score}")

        # Вставляем ваш код для вывода расписания сотрудников:
        print("\nСоставленное расписание:")
        for emp in employees:
            print(f"\nСотрудник {emp['id']} ({emp['role']}):")
            tasks = []
            for epic_name, epic_tasks in final_schedule.items():
                for task in epic_tasks:
                    if task['employee_id'] == emp['id']:
                        start_day = task['start_day']
                        duration = task['duration']
                        end_day = start_day + duration
                        tasks.append((start_day, end_day, epic_name, task['role']))
            for task in sorted(tasks):
                print(f"  Эпик {task[2]} ({task[3]}): с {task[0]} до {task[1]}")

        # Визуализация расписания
        visualize_schedule(final_schedule)
    else:
        print("Не удалось сформировать итоговое расписание.")

if __name__ == "__main__":
    main()
