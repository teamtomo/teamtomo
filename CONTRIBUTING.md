# Contribution Guidelines for TeamTomo

Welcome to TeamTomo, and thanks for your interest in contributing!

The following document is a set of guidelines for contributing to TeamTomo and its packages which are hosted through the [TeamTomo Organization](https://github.com/teamtomo) on GitHub. TeamTomo is a open, volunteer-based consortium of scientists and researchers contributing core infrastructure for cryo-EM/ET computational methods development.

We aim to build up community, not barriers, so these are not strict rules. Please feel free to propose changes to this document through a pull request.

## Table of Contents

* [Scope of TeamTomo](#scope-of-teamtomo)
* [Join our Zulip chat for open discussions](#join-our-zulip-chat-for-open-discussions)
* [Migrating an existing package to TeamTomo](#migrating-an-existing-package-to-teamtomo)
* [Adding a new package to TeamTomo](#adding-a-new-package-to-teamtomo)
* [Pull requests and review processes](#pull-requests-and-review-processes)
* [Feature requests and additions](#feature-requests-and-additions)

🚧 More topics and guides to come! 🚧

### Scope of TeamTomo

TeamTomo is common cryo-EM/ET computational infrastructure built by developers, for developers. The scope of TeamTomo includes image processing routines specific to cryo-EM/ET (primitives), such as CTF calculations, as well as more complex data processing steps (algorithms), such as motion estimation from raw movies. TeamTomo also supports I/O package for handling data specific to cryo-EM/ET.

Note that the scope of TeamTomo does not extend to fully-fledged programs like tomogram reconstruction program which takes in unaligned tilt frames, some metadata file, and outputs a volume. We want to enable scientists with the _individual computational pieces_ (primitives/algorithms/io) for building exciting tools and programs for cryo-EM/ET rather than re-inventing data processing pipelines.

If you've built something cool using TeamTomo, please let us know! We'd love to assemble a showcase of research that open science and common infrastructure enables. If you have some functionality to contribute to TeamTomo or an idea to add, please continue reading.

### Join our Zulip chat for open discussions

TeamTomo uses [Zulip](https://zulip.com) as a messaging service for code questions, open discussions, and project development planning. If you want to start contributing to TeamTomo, [our Zulip channel](https://imagesc.zulipchat.com/#narrow/channel/426493-TeamTomo) is the best place to start.

[imagesc.zulipchat.com/#narrow/channel/426493-TeamTomo](https://imagesc.zulipchat.com/#narrow/channel/426493-TeamTomo)

### Migrating an existing package to TeamTomo

If you've developed a package you think is in-scope for TeamTomo, please follow the [existing package migration](notes/migrate-existing-repo.md) guide and open a pull request to add a new package into the monorepo.

_Note: if this is an I/O package, please reach out on Zulip for creating a new repo under the organization._

### Adding a new package to TeamTomo

If you're looking to add a new package to TeamTomo, again the first place to start is the Zulip channel. Once there's a clear plan in place, follow the guide [creating a new TeamTomo package](notes/create-new-package.md) and open a new pull request

### Pull requests and review processes

🚧 Under construction 🚧

### Feature requests and additions

🚧 Under construction 🚧
