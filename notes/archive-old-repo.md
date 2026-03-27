# Archiving Legacy Repositories

This document provides a checklist for archiving individual package repositories as they are migrated into the teamtomo monorepo.

## Pre-Archive Checklist

For each package repository being archived (e.g., `teamtomo/membrain-pick`, `teamtomo/fidder`), complete the following steps:

### 1. Sync Latest Changes to Monorepo

- [ ] Pull latest changes from the standalone repository
- [ ] Compare with monorepo version at `packages/<package-name>/`
- [ ] Port any new commits/features that aren't in the monorepo yet (follow [existing migration guide](migrate-existing-repo.md) if needed)
- [ ] Ensure version numbers are in sync
- [ ] Update `CHANGELOG.md` in monorepo (if applicable)
- [ ] Create PR in monorepo with synced changes (if any, see below)

### 2. Migrate Open Issues

For each open issue in the legacy repository:

- [ ] Review if the issue is still relevant
- [ ] Create equivalent issue in the monorepo using this template:

  ```markdown
  **Migrated from**: [package-name#<issue-number>](link-to-original-issue)
  **Package**: `<package-name>`
  
  [Original issue content]
  
  ---
  *Original discussion and context can be found in the linked issue above.*
  ```

- [ ] Add appropriate labels: `migrated`, `<package-name>`
- [ ] Reference the new issue in the old issue:
  
  ```markdown
  This issue has been migrated to the teamtomo monorepo: teamtomo/teamtomo#<new-issue-number>
  ```

- [ ] Close the old issue

### 3. Handle Open Pull Requests

For each open PR in the legacy repository:

**Option A: Merge if ready**

- [ ] Review and merge if approved and CI passes
- [ ] Sync merged changes to monorepo (see step 1)

**Option B: Migrate if still needed**

- [ ] Create equivalent PR in monorepo targeting the package subdirectory
- [ ] Reference original PR author and give credit
- [ ] Add comment to original PR:

```markdown
This PR has been recreated in the monorepo: teamtomo/teamtomo#<new-pr-number>. Development of TeamTomo packages will continue in the monorepo.
```

- [ ] Close the original PR

**Option C: Close if obsolete**

- [ ] Add comment explaining why (e.g., "This is now handled differently in the monorepo")
- [ ] Thank the contributor
- [ ] Close the PR

### 4. Update Repository README

Add a prominent notice at the top of the legacy repository's `README.md`:

````markdown
> **⚠️ This repository has been archived**
>
> Development of `<package-name>` has moved to the [teamtomo monorepo](https://github.com/teamtomo/teamtomo).
>
> - **New issues**: Please open in [teamtomo/teamtomo](https://github.com/teamtomo/teamtomo/issues)
> - **Installation**: `uv pip install <package-name>` or see [installation guide](https://github.com/teamtomo/teamtomo#installation)
> - **Package location**: [`packages/<package-name>/`](https://github.com/teamtomo/teamtomo/tree/main/packages/<package-name>)
> - **Contributing**: See the [monorepo contributing guide](https://github.com/teamtomo/teamtomo/blob/main/CONTRIBUTING.md)
>
> This repository remains available for historical reference but is no longer maintained.