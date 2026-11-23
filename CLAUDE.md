# CLAUDE.md - cách làm việc trong repo vi-rag-agent

Repo portfolio của Dương Minh Quang (xin thực tập/fresher AI Engineer). Tiếng Việt khi trao đổi; code, identifier,
commit, README, docs trong repo bằng tiếng Anh. Kế hoạch: `docs/ROADMAP.md`. Quy trình đánh giá: `docs/EVALUATION.md`.

## Lệnh thường dùng
```powershell
pip install -e ".[dev]"
ruff check . ; ruff format --check . ; pytest
```

## Bất biến
1. Không bịa số liệu. Số trong README/CHANGELOG chỉ lấy từ `results/*.json` của lần chạy thật. Chưa đo thì ghi "not measured".
2. Không rò rỉ dữ liệu: chia train/dev/test một lần, cố định seed, chỉ tune trên dev.
3. Không commit dữ liệu, `.env`, token, checkpoint, index. Không dùng nhãn đáp án công khai để "tái hiện" kết quả.
4. Mọi hàm đo (metric) có test so với giá trị tính tay. Sửa logic thì chạy lại test.
5. Không sao chép code người khác; ghi nguồn và license của mọi dataset/model.
6. Nói rõ điều gì chưa kiểm chứng.

## Chất lượng: không mặc định là đúng
Trước khi báo "xong" một việc, Claude phải làm đủ các bước sau và ghi kết quả trong PR:
1. **Recheck**: chạy lại lint, test; tính lại mọi con số từ file kết quả; đọc lại diff như một reviewer.
2. **Phản biện**: nêu ít nhất 3 câu hỏi khó mà người phỏng vấn/reviewer có thể hỏi (rò rỉ dữ liệu, baseline, cỡ mẫu, so sánh có công bằng không) và trả lời bằng bằng chứng hoặc ghi "chưa kiểm chứng".
3. **Thử phá**: với code mới, thử đột biến (mutation) hoặc ca biên để xem test có thật sự bắt được lỗi.
4. **Gợi ý**: đề xuất ít nhất 2 hướng cải thiện kèm chi phí/lợi ích; nêu lựa chọn thay thế đã cân nhắc.
5. **Không đồng ý khi cần**: nếu yêu cầu của người dùng có rủi ro hoặc có cách tốt hơn, nói rõ trước khi làm.

## README luôn đúng hiện trạng
README.md là trang đầu tiên người tuyển dụng đọc trên GitHub. Trong mỗi PR, Claude tự hỏi: "README còn đúng không?"
và cập nhật trong cùng PR khi PR đó:
- thêm kết quả mới vào `results/` (bảng Results, chỉ chép số từ JSON, có link file và report);
- đóng milestone/release (dòng **Status** ghi đúng phiên bản mới nhất và milestone đang làm);
- thêm hoặc đổi extra, lệnh chạy, script, workflow, cấu trúc thư mục, cách cài đặt (Quick start, Project layout);
- đổi quyết định (ADR) làm thay đổi kiến trúc hoặc hệ thống được chọn.
Nếu không cần sửa, ghi "README: no change needed (lý do)" trong phần Checks của PR.
`tests/test_readme.py` chặn lỗi cơ bản (Status khớp CHANGELOG, đủ extras, link không gãy); test không thay thế việc đọc lại README.

## Quy trình (ADR-0004)
Kế hoạch: `PLAN.md`. Quyết định: `docs/decisions/`. Giao thức thí nghiệm: `docs/protocols/` (merge vào `main` trước khi chạy).
Cổng tái tạo: script thí nghiệm mới phải tái tạo số tham chiếu của milestone trước, lệch thì dừng.
Test chỉ chạy một lần khi đóng milestone, cho cấu hình đã chọn trên validation (ADR-0001).

## Quy trình GitHub
Milestone -> issue (tiêu chí xong) -> nhánh `issue-<n>-<chủ-đề>` (job GPU: `kaggle/<job>`) -> commit `<type>(scope): ... (#n)` -> PR (`Closes/Refs #n`, Summary · Changes · Evidence/Checks · Limitations) -> tự review + review đối lập nếu có ADR/giao thức -> squash-merge.
Commit do Duong Minh Quang làm tác giả, không Co-Authored-By; PR/issue không có footer "generated".
Đóng milestone: viết `docs/reports/vX.Y.0.md`, chuyển CHANGELOG, mở PR `chore(release): vX.Y.Z`; workflow `release` tự tạo tag + Release (xem `docs/reports/README.md`).
Commit theo Conventional Commits. Không push thẳng vào `main`. Lịch sử git trung thực, không backdate.
