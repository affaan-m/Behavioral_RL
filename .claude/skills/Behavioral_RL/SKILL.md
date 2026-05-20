```markdown
# Behavioral_RL Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and workflows used in the `Behavioral_RL` repository, a TypeScript codebase focused on behavioral reinforcement learning. It covers file organization, coding conventions, import/export styles, and automated workflows for dependency management. The guide also includes best practices for writing and running tests.

## Coding Conventions

**File Naming**
- Use camelCase for file names.
  - Example: `behavioralAgent.ts`, `rewardFunction.ts`

**Import Style**
- Use relative imports for modules within the project.
  - Example:
    ```typescript
    import { computeReward } from './rewardFunction';
    ```

**Export Style**
- Use named exports for functions, classes, and constants.
  - Example:
    ```typescript
    // In rewardFunction.ts
    export function computeReward(state: State): number { ... }
    ```

**Commit Messages**
- Freeform style, no strict prefixing.
- Average commit message length: ~55 characters.

## Workflows

### Bulk Dependency Update Across Experiment Directories
**Trigger:** When you need to update core Python dependencies for all experiment runs managed by wandb.
**Command:** `/bulk-update-dependencies`

1. Identify all wandb run directories under `wandb/*/files/`.
2. For each run directory, open the `requirements.txt` file.
3. Update the following dependencies to their new versions: `idna`, `mistune`, `ujson`, `urllib3`.
4. Save the updated `requirements.txt` in each directory.
5. Commit all updated `requirements.txt` files in a single commit.

**Example Directory Structure:**
```
wandb/
  run1/
    files/
      requirements.txt
  run2/
    files/
      requirements.txt
```

**Example Update in `requirements.txt`:**
```
idna==3.4
mistune==2.0.4
ujson==5.8.0
urllib3==2.0.2
```

## Testing Patterns

- Test files follow the pattern: `*.test.*`
  - Example: `behavioralAgent.test.ts`
- The specific testing framework is not detected; check the project for further details.
- Place test files alongside or near the code they test.

**Example Test File:**
```typescript
// behavioralAgent.test.ts
import { computeReward } from './rewardFunction';

test('computeReward returns correct value', () => {
  expect(computeReward({ score: 10 })).toBe(100);
});
```

## Commands

| Command                   | Purpose                                                         |
|---------------------------|-----------------------------------------------------------------|
| /bulk-update-dependencies | Update Python dependencies in all wandb experiment requirements. |

```